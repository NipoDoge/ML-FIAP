# Tech Challenge — Plataforma MLOps (Churn + Recomendação)

Pipeline end-to-end: **FastAPI** · **PyTorch** · **scikit-learn** · **MLflow** · **Airflow** · **PostgreSQL** · **Docker**.

- **TC01:** classificação de churn (tabular + MLP)
- **TC02:** recomendação user-item (embeddings PyTorch)

## Documentação

**Toda a documentação oficial está em:**

**[docs/DOCUMENTACAO.md](docs/DOCUMENTACAO.md)**

(arquitectura, setup, testes, model cards, demo banca, troubleshooting)

## Início rápido

```bash
cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
make install-dev
make docker-fresh
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
```

| Interface | URL | Credenciais |
|-----------|-----|-------------|
| Swagger | http://localhost:8000/docs | `admin@admin.com` / `admin1` |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5000 | — |

## Testes e validação

**Roteiro detalhado TC02 (stack zero → DVC → Swagger → Registry):** [`docs/roteiro_teste_completo.md`](docs/roteiro_teste_completo.md)

```bash
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

## Vídeo

- STAR: https://youtu.be/h5Fn-4741No
