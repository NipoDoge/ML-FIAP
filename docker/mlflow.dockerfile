FROM ghcr.io/mlflow/mlflow:v3.10.1

# Backend store PostgreSQL (bases airflow / processing / mlflow no db_processing).
# API/worker tambem usam mlflow==3.10.1; manter cliente e servidor alinhados evita
# chamadas a endpoints inexistentes em versoes antigas (ex.: logged-models).
RUN pip install --no-cache-dir psycopg2-binary
