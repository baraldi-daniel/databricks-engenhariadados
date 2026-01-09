import dlt
from pyspark.sql.functions import *

# =========================================================================
# CONFIGURAÇÃO GERAL
# =========================================================================
# Ajuste para seu volume
INPUT_PATH = "/Volumes/catalogo_pipeline/schema_pipeline/volume"

# SEU MODELO ESPECÍFICO
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Schema Tolerante (7 colunas, mas aceita 6 via PERMISSIVE)
schema_bronze = "customer_bk STRING, customer_name STRING, birth_date STRING, segment STRING, region STRING, effective_ts STRING, change_type STRING"

# =========================================================================
# 1. BRONZE (Ingestão Raw)
# =========================================================================
@dlt.table(
    name="customer_bronze",
    comment="Ingestão Raw. Usa mode PERMISSIVE para aceitar arquivos sem a flag change_type."
)
def customer_bronze():
    return (
        spark.readStream
            .format("csv")
            .schema(schema_bronze)
            .option("header", "true")
            .option("sep", ",")
            .option("mode", "PERMISSIVE") 
            .load(INPUT_PATH)
    )

# =========================================================================
# 2. SILVER PREP (Limpeza com Llama 3.3)
# =========================================================================
@dlt.view(name="customer_silver_prep_v")
def customer_silver_prep_v():
    df = dlt.read_stream("customer_bronze")
    
    # 1. Lógica Upsert/Delete (Management by Exception)
    df = df.withColumn("change_type", coalesce(col("change_type"), lit("UPSERT")))

    # 2. Limpeza Determinística (Regex e Datas)
    df = df.withColumn("customer_bk_clean", regexp_replace(col("customer_bk"), "[^0-9]", "")) \
           .withColumn("effective_ts_parsed", 
                coalesce(
                    to_timestamp(col("effective_ts")), 
                    to_timestamp(col("effective_ts"), "yyyy-MM-dd HH:mm:ss"),
                    to_timestamp(col("effective_ts"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
                    to_timestamp(col("effective_ts"), "dd/MM/yyyy")
                )
            )

    # --- 3. LIMPEZA COM IA (AI_QUERY) ---
    
    # A. Criação dos Prompts (Instruções para o Llama 3.3)
    # Dica: O Llama 3.3 obedece bem a "SYSTEM PROMPT" ou instruções diretas.
    
    # Prompt para REGIÃO
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

    # Prompt para SEGMENTO
    df = df.withColumn("prompt_segment", 
        concat(
            lit("Padronize o segmento comercial para: 'Enterprise', 'SMB' ou 'Varejo'. "),
            lit("Corrija erros de digitação e capitalização. "),
            lit("Responda APENAS a palavra da categoria correta. Sem explicações. Entrada: "),
            col("segment")
        )
    )

    # B. Chamada ao Modelo (ai_query)
    # Usamos expr() para injetar a função SQL dentro do Python
    df = df.withColumn("region_clean", expr(f"ai_query('{MODEL_ENDPOINT}', prompt_region)")) \
           .withColumn("segment_clean", expr(f"ai_query('{MODEL_ENDPOINT}', prompt_segment)")) \
           .withColumn("customer_name_clean", trim(col("customer_name")))

    # --- 4. Watermark ---
    return df.withWatermark("effective_ts_parsed", "1 hour") \
             .dropDuplicates(["customer_bk_clean", "effective_ts_parsed"])

# =========================================================================
# 3. SILVER HISTORY (CDC / SCD2)
# =========================================================================
dlt.create_streaming_table(
    name="dim_customer_silver_history",
    comment="Histórico SCD2 com dados limpos via GenAI."
)

dlt.apply_changes(
    target = "dim_customer_silver_history",
    source = "customer_silver_prep_v",
    keys = ["customer_bk_clean"],
    sequence_by = col("effective_ts_parsed"),
    stored_as_scd_type = 2,
    apply_as_deletes = expr("change_type = 'DELETE'"),
    
    # Removemos as colunas sujas e os prompts da tabela final
    except_column_list = ["customer_bk", "effective_ts", "change_type", "region", "segment", "prompt_region", "prompt_segment"]
)

# =========================================================================
# 4. GOLD CURRENT (Snapshot)
# =========================================================================
@dlt.table(
    name="dim_customer_gold_current",
    comment="Visão final apenas com clientes ativos e dados normalizados."
)
def dim_customer_gold_current():
    return (
        dlt.read("dim_customer_silver_history")
           .filter(col("__END_AT").isNull())
           .drop("__START_AT", "__END_AT")
    )