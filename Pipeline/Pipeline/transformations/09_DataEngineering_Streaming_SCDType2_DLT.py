import dlt
from pyspark.sql.functions import *

# =========================================================================
# GENERAL CONFIG
# =========================================================================
# INPUT PATH: Direct ADLS Gen2 path
# Format: abfss://<container>@<storage_account>.dfs.core.windows.net/<folder>
INPUT_PATH = "abfss://container@baraldistorage.dfs.core.windows.net/data_medallion/sdp/customer/"

# LLM MODEL
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Fault Tolerant Schema 
# Note: 'change_type' is crucial for the LGPD deletion logic
schema_bronze = "customer_bk STRING, customer_name STRING, birth_date STRING, segment STRING, region STRING, effective_ts STRING, change_type STRING"

# =========================================================================
# 1. BRONZE (Raw Ingestion - Auto Loader)
# =========================================================================
@dlt.table(
    name="customer_bronze",
    comment="Raw Ingestion using Auto Loader (cloudFiles) for real-time detection."
)
# EXPECT: Just a warning. We want to ingest everything to see what is failing in the source.
@dlt.expect("warn_valid_bk_raw", "customer_bk IS NOT NULL") 
def customer_bronze():
    return (
        spark.readStream
            .format("cloudFiles")               # Activates Auto Loader
            .option("cloudFiles.format", "csv") # Specifies the source format is CSV
            .schema(schema_bronze)              # Enforces the schema
            .option("header", "true")
            .option("sep", ",")
            # Auto Loader automatically handles schema evolution and rescued data
            .load(INPUT_PATH)
    )

# =========================================================================
# 2. SILVER FIRST STEP (Clean with Llama 3.3 & Validation)
# =========================================================================
@dlt.view(name="customer_silver_prep_v")
# EXPECT OR DROP: Critical for SCD2. If PK or Date is missing, we cannot history-track it.
@dlt.expect_or_drop("valid_pk_clean", "customer_bk_clean IS NOT NULL AND length(customer_bk_clean) > 0")
@dlt.expect_or_drop("valid_timestamp", "effective_ts_parsed IS NOT NULL")
# EXPECT: AI Monitoring. If Llama hallucinations occur, we warn but keep the row to fix later.
@dlt.expect("ai_region_is_valid_uf", "length(region_clean) == 2") 
@dlt.expect("ai_segment_in_list", "segment_clean IN ('Enterprise', 'SMB', 'Varejo')")
def customer_silver_prep_v():
    df = dlt.read_stream("customer_bronze")
    
    # 1. Upsert/Delete Management
    # If change_type is null, assume it's a standard UPSERT. 
    df = df.withColumn("change_type", coalesce(col("change_type"), lit("UPSERT")))

    # 2. Deterministic Cleaning (Regex and Dates)
    df = df.withColumn("customer_bk_clean", regexp_replace(col("customer_bk"), "[^0-9]", "")) \
           .withColumn("effective_ts_parsed", 
                coalesce(
                    to_timestamp(col("effective_ts")), 
                    to_timestamp(col("effective_ts"), "yyyy-MM-dd HH:mm:ss"),
                    to_timestamp(col("effective_ts"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
                    to_timestamp(col("effective_ts"), "dd/MM/yyyy")
                )
            )

    # 3. CLEAN WITH AI (AI_QUERY)
    
    # 3.1 Prompt Creation 
    
    # Prompt for REGION
    df = df.withColumn("prompt_region", 
        concat(
            lit("Você é um especialista em geografia do Brasil. Sua tarefa é padronizar nomes de locais para a SIGLA UF (2 letras). "),
            lit("Regras: 1. Se for 'São Paulo', 'sao paulo', 'sp', devolva 'SP'. "),
            lit("2. Se for 'Minas Gerais', 'mg', devolva 'MG'. "),
            lit("3. Se for 'Bahia', 'ba', devolva 'BA'. "),
            lit("4. Se não for um estado brasileiro óbvio, devolva 'OUTROS'. "),
            lit("IMPORTANTE: Responda APENAS a sigla de 2 letras. Nada mais. Entrada: "),
            col("region")
        )
    )

    # Prompt for SEGMENT
    df = df.withColumn("prompt_segment", 
        concat(
            lit("Padronize o segmento comercial para: 'Enterprise', 'SMB' ou 'Varejo'. "),
            lit("Corrija erros de digitação e capitalização. "),
            lit("Responda APENAS a palavra da categoria correta. Sem explicações. Entrada: "),
            col("segment")
        )
    )

    # 3.2 Call Model (ai_query)
    df = df.withColumn("region_clean", expr(f"ai_query('{MODEL_ENDPOINT}', prompt_region)")) \
           .withColumn("segment_clean", expr(f"ai_query('{MODEL_ENDPOINT}', prompt_segment)")) \
           .withColumn("customer_name_clean", trim(col("customer_name")))

    # 4. Watermark & Deduplication
    # Ensures idempotency if the same file is processed twice or duplicates exist in batch
    return df.withWatermark("effective_ts_parsed", "1 hour") \
             .dropDuplicates(["customer_bk_clean", "effective_ts_parsed"])

# =========================================================================
# 3. SILVER HISTORY (CDC / SCD2)
# =========================================================================
dlt.create_streaming_table(
    name="dim_customer_silver_history",
    comment="SCD Type 2 History. Handles UPSERTs and DELETEs (LGPD)."
)

dlt.apply_changes(
    target = "dim_customer_silver_history",
    source = "customer_silver_prep_v",
    keys = ["customer_bk_clean"],
    sequence_by = col("effective_ts_parsed"),
    stored_as_scd_type = 2,
    
    # LOGIC FOR LGPD / DELETION
    apply_as_deletes = expr("change_type = 'DELETE'"),
    
    # Exclude technical/dirty columns from the final table
    except_column_list = ["customer_bk", "effective_ts", "change_type", "region", "segment", "prompt_region", "prompt_segment"]
)

# =========================================================================
# 4. GOLD CURRENT (Snapshot)
# =========================================================================
@dlt.table(
    name="dim_customer_gold_current",
    comment="Final view: Active customers only."
)
# Final sanity check for Gold data
@dlt.expect("gold_quality_check", "customer_name_clean IS NOT NULL")
def dim_customer_gold_current():
    return (
        dlt.read("dim_customer_silver_history")
           .filter(col("__END_AT").isNull()) # Filter only currently active records
           .drop("__START_AT", "__END_AT")
    )