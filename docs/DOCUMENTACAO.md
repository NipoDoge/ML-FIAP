# Documentação oficial — Plataforma MLOps (TC01 + TC02)

Documento **único** do repositório: arquitectura, operação, testes, model cards e entrega à banca.

**Stack:** FastAPI · PyTorch · scikit-learn · MLflow · Airflow · PostgreSQL · Docker  
**Domínios:** `churn` (tabular + MLP) · `recommendation` (PyTorch embeddings)  
**Rotas:** `/v1/domains/{domain}/…` · **DAG:** `ml_training_dispatch`

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Arquitectura](#2-arquitectura)
3. [Contrato da plataforma](#3-contrato-da-plataforma)
4. [Mapa do repositório](#4-mapa-do-repositório)
5. [Setup e execução](#5-setup-e-execução)
6. [Testes e validação](#6-testes-e-validação) — ver também [`roteiro_teste_completo.md`](roteiro_teste_completo.md) (TC02 passo a passo)
7. [Apresentação à banca](#7-apresentação-à-banca)
8. [Model Card — Churn (TC01)](#8-model-card--churn-tc01)
9. [Model Card — Recomendação (TC02)](#9-model-card--recomendação-tc02)
10. [MLP PyTorch (churn)](#10-mlp-pytorch-churn)
11. [Observabilidade e manutenção](#11-observabilidade-e-manutenção)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)

---

## 1. Visão geral

### Problema e objectivo

- **TC01 (churn):** operadora telecom precisa identificar clientes em risco de cancelamento para acção de retenção. Pipeline: baseline → feature engineering → comparação sklearn + **MLP PyTorch** → API autenticada.
- **TC02 (recomendação):** ranquear itens para `user_id` com embeddings PyTorch (MovieLens como proxy e-commerce).

### O que a plataforma faz

- Ciclo MLOps completo: **treinar → registar → promover → predizer → rollback**
- Dois domínios na mesma API, mesmas rotas de ciclo de vida, payloads diferentes
- Orquestração Airflow (`ml_training_dispatch`), tracking MLflow (Postgres + artefactos)
- JWT + roles (admin vs utilizador)
- Drift PSI e latência **offline** (scripts de manutenção)

### O que não faz

- Regressão / multiclasse (churn é binário)
- Estimativa de LTV ou causalidade
- Drift ou alertas em tempo real na API
- SMOTE / undersampling

### Vídeos

- STAR: https://youtu.be/UxTcH5kxdwc
- Easy-run: https://youtu.be/_9VqWvDYOuk

---

## 2. Arquitectura

### 2.1 Ciclo de produto (imutável)

```text
TREINAR  →  REGISTRAR  →  PROMOVER  →  PREDIZER
```

Novos domínios **plugam** na plataforma; login, tabelas BD e rotas de ciclo de vida **não redesenham**.

### 2.2 Anéis (`_ring`)

| Anel | Pasta | Responsabilidade |
|------|-------|------------------|
| **0 — infra** | `src/core/` | Settings, logging, Postgres |
| **1 — platform** | `src/platform_ring/`, `src/services/processor/` | HTTP, auth, runs, promote, predict |
| **2 — core + domains** | `src/ml_core_ring/`, `src/domains/` | Registries; plugin por domínio |
| **3 — executors** | `src/executors_ring/` | Algoritmos, pipelines, worker HTTP |
| **4 — orchestration** | `src/orchestration_ring/`, `airflow/dags/` | Airflow, conf merge, persistência |

**Princípio:** a API **não treina** modelos pesados inline; delega ao Airflow (→ worker ou executor tabular).

**Regras de import (resumo):** `executors_ring` não importa API; `platform_ring` não importa algoritmos directamente — usa registries em `ml_core_ring`.

### 2.3 Topologia Docker

```text
Cliente → api_processing :8000
              ├── /v1/domains/churn/…        → tabular + MLP
              └── /v1/domains/recommendation/… → worker_recommendation :8010

Airflow ml_training_dispatch
              ├── domain=churn      → orchestration_ring/tabular_training
              └── domain=recommendation → worker POST /train

Runtime /predict → deployed_models (Postgres `processing`)
Promote          → deployed_models + MLflow Registry (alias Production, best-effort)
```

| Serviço | Porta | Função |
|---------|-------|--------|
| `api_processing` | 8000 | API FastAPI |
| `worker_recommendation` | 8010 | Treino reco |
| `airflow_webserver` | 8080 | UI Airflow |
| `mlflow_server` | 5000 | Tracking + Registry |
| `database_processing` | 5432 | Postgres (3 bases) |
| `pgadmin_db` | 5050 | pgAdmin |
| `dozzle_logs` | 8888 | Logs contentores |

### 2.4 Postgres — 3 bases

```text
db_processing (:5432)
├── processing   → users, pipeline_runs, deployed_models, predictions
├── airflow      → metadados Airflow
└── mlflow       → tracking + Model Registry
```

### 2.5 MLflow

| Camada | Onde |
|--------|------|
| Metadados | Postgres `mlflow` |
| Tracking URI | `http://mlflow_server:5000` (compose) / `http://localhost:5000` (host) |
| Artefactos | `./src/artifacts/mlruns` ↔ `/mlflow/artifacts` nos contentores |
| Runtime `/predict` | **`deployed_models`** (Postgres) — **não** o Registry |
| Registry no promote | side-effect → `tc02_recommender`, `churn_fe_model` (alias Production) |

### 2.6 DAG `ml_training_dispatch`

Uma DAG canónica com **branch por `domain`**:

```text
validate_dispatch → branch
  ├── tabular (churn): baseline → FE → promote opcional → notify
  └── recommendation: worker /train → notify
```

**Conf:** Variable `ml_training_dispatch_conf` ∪ `dag_run.conf` (conf do run sobrescreve defaults).

| Campo | Tabular | Recomendação |
|-------|---------|--------------|
| `domain` | `churn` | `recommendation` |
| `csv_path` | obrigatório | ignorado |
| `auto_promote` | opcional (FE) | N/A (promote via API) |
| `params`, `top_k` | N/A | worker |

---

## 3. Contrato da plataforma

### 3.1 Rotas globais

| Método | Rota | Auth |
|--------|------|------|
| POST | `/v1/auth/authenticate` | — |
| GET | `/v1/health` | — |
| GET/POST | `/v1/users`, `/v1/roles` | User |

### 3.2 Rotas por domínio

Prefixo: **`/v1/domains/{domain}/`** — domínios actuais: `churn`, `recommendation`.

| Operação | Rota | Auth |
|----------|------|------|
| Predict | `POST …/predict` | User |
| Runs | `GET …/admin/runs` | Admin |
| Promote | `POST …/admin/promote` | Admin |
| Rollback | `POST …/admin/rollback` | Admin |
| Deploy history | `GET …/admin/deployments/history` | Admin |
| Treino Airflow | `POST …/admin/train/trigger` | Admin |
| Treino sync (churn) | `…/admin/train/baseline`, `…/admin/train/feature-engineering` | Admin |
| Treino sync (reco) | `POST …/admin/train/sync` | Admin |

Swagger: tags **`domain-churn`** e **`domain-recommendation`**.

**Extensão:** novo `{domain}` + schema Pydantic de `/predict` — **sem** novas rotas de ciclo de vida.

### 3.3 Modelo de dados

| Tabela | Papel |
|--------|--------|
| `users` / `roles` | Identidade |
| `pipeline_runs` | Todo treino |
| `deployed_models` | 1 activo por `domain` |
| `predictions` | Auditoria de inferência |

### 3.4 Credenciais (teste)

| Serviço | Login | Senha |
|---------|-------|-------|
| API / Swagger | `admin@admin.com` | `admin1` |
| pgAdmin | `admin@admin.com` | `admin1` |
| Postgres | `admin` | `admin1` |
| Airflow | `airflow` | `airflow` |

### 3.5 Novo domínio (checklist)

1. `domains/<nome>/plugin.py` — `DomainPlugin`
2. `executors_ring/<tipo>/` — `TrainBackend`
3. Registo em `ml_core_ring` (train + inference)
4. Branch ou task em `ml_training_dispatch`
5. `platform_ring/domains/<nome>/router.py`
6. Schema em `platform_ring/schemas/`

---

## 4. Mapa do repositório

| TC | Rotas HTTP | ML / treino | Dados |
|----|------------|-------------|-------|
| **Churn** | `src/platform_ring/domains/churn/` | `services/pipelines/`, `executors_ring/tabular_classification/` | `ml_data/uploads/`, `src/data/` |
| **Reco** | `src/platform_ring/domains/recommendation/` | `domains/recommendation/`, worker `:8010` | `data/recommendation/`, `models/recommendation/` |

| Variável / path | Valor |
|-----------------|-------|
| Uploads CSV | `ml_data/uploads/` (`ML_SHARED_PATH`) |
| Artefactos MLflow | `src/artifacts/mlruns` |
| Conf Airflow | `airflow/bootstrap/ml_training_dispatch_conf.json` |
| Schemas HTTP | `src/platform_ring/schemas/` |
| Script E2E | `scripts/validate_platform.sh` |

---

## 5. Setup e execução

### 5.1 Pré-requisitos

- Docker + Compose ≥ 2.20
- Python ≥ 3.10 (pytest no host)
- Portas livres: 8000, 8080, 5000, 8010, 5432, 5050, 8888

### 5.2 Primeira execução

```bash
cd /home/gabriel/Machine-Learning
cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
make install-dev
make docker-fresh
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
```

### 5.3 Variáveis `.env` essenciais

| Variável | Valor típico |
|----------|--------------|
| `ENVIRONMENT` | `development` |
| `DATABASE_NAME` | `processing` |
| `AIRFLOW_DATABASE_NAME` | `airflow` |
| `MLFLOW_DATABASE_NAME` | `mlflow` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `ML_SHARED_PATH` | `ml_data/uploads` |
| `USE_MLP_FOR_PREDICTION` | `TRUE` |

Dados: `ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv`, `data/recommendation/ratings.csv`.

### 5.4 Fluxo manual (API)

1. Login → `POST /v1/auth/authenticate`
2. Treino (dev): baseline + FE (churn) ou sync (reco)
3. Promote → `POST …/admin/promote`
4. Predict → `POST …/predict`
5. Rollback (opcional) → `POST …/admin/rollback`

Em `ENVIRONMENT=production`, treino síncrono retorna **403** — usar Airflow.

### 5.5 Interfaces web

| Serviço | URL |
|---------|-----|
| Swagger | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| pgAdmin | http://localhost:5050 (host: `database_processing`) |
| Dozzle | http://localhost:8888 |

### 5.6 Token + exemplos curl

```bash
export API=http://localhost:8000
TOKEN=$(curl -s -X POST "$API/v1/auth/authenticate" \
  -F username=admin@admin.com -F password=admin1 \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

# Churn predict
curl -s -X POST "$API/v1/domains/churn/predict" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"gender":"Female","seniorcitizen":0,"partner":1,"dependents":0,"tenure":12,"phoneservice":1,"multiplelines":0,"internetservice":"DSL","onlinesecurity":0,"onlinebackup":0,"deviceprotection":0,"techsupport":0,"streamingtv":0,"streamingmovies":0,"contract":"Month-to-month","paperlessbilling":1,"paymentmethod":"Electronic check","monthlycharges":70.5,"totalcharges":845.0}' \
  | python3 -m json.tool

# Recommendation predict
curl -s -X POST "$API/v1/domains/recommendation/predict" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_id":1,"top_k":5}' | python3 -m json.tool
```

---

## 6. Testes e validação

> **Roteiro passo a passo TC02** (docker-fresh, DVC, Swagger, Model Registry): [`roteiro_teste_completo.md`](roteiro_teste_completo.md)

### 6.1 Unitários (anéis)

```bash
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
```

> Não usar `make test-fast` como gate — suite completa pode falhar em `tests/src/` (pandera × numpy 2.0).

### 6.2 E2E plataforma (~15–20 min)

```bash
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

Esperado: **46+ PASS · 0 FAIL**. Cobre infra, auth, worker, Airflow DAG, promote, predict, rollback, MLflow Registry.

Modos: `--infra-only`, `--skip-dag`, `--skip-build`.

### 6.3 Churn manual

```bash
# Treino sync (rápido)
curl -s -X POST "$API/v1/domains/churn/admin/train/baseline" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv"

curl -s -X POST "$API/v1/domains/churn/admin/train/feature-engineering" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv"

curl -s -X POST "$API/v1/domains/churn/admin/promote" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Pré-requisito churn MLP: `USE_MLP_FOR_PREDICTION=TRUE`.

### 6.4 Smoke rápido

```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
curl -s http://localhost:8010/health
docker exec airflow_scheduler airflow dags list | grep ml_training_dispatch
```

### 6.5 Checklist pré-banca

- [ ] Stack Up (`docker compose ps`)
- [ ] pytest anéis verde
- [ ] `validate_platform.sh` 0 FAIL
- [ ] Churn predict manual OK
- [ ] MLflow UI com modelos visíveis

---

## 7. Apresentação à banca

### Mensagem central (20 s)

> Plataforma MLOps com **dois domínios**, arquitectura em **anéis**, API `/v1/domains/{domain}/…`, Airflow **`ml_training_dispatch`**, MLflow unificado, promote/rollback em Postgres e Registry como side-effect.

### Roteiro STAR para vídeo de até 5 min

| Tempo | STAR | Cena / evidência | Fala sugerida |
|-------|------|------------------|---------------|
| 0:00–0:55 | **S — Situação** | README, Swagger ou topo da documentação | "O projeto nasceu de dois desafios de Machine Learning Engineering. No TC01, o problema era prever churn em telecom para apoiar ações de retenção. No TC02, o desafio evoluiu para recomendação user-item, usando MovieLens como proxy de e-commerce. O ponto comum entre os dois não era só treinar modelos, mas colocar esses modelos num fluxo operacional confiável." |
| 0:55–1:35 | **T — Tarefa** | Mostrar `/v1/domains/churn` e `/v1/domains/recommendation` no Swagger | "A tarefa foi transformar esses casos em uma plataforma MLOps multi-domínio. Ela precisava manter o mesmo ciclo para problemas diferentes: treinar, registrar, promover, predizer e fazer rollback. Também precisava ter API autenticada, orquestração no Airflow, tracking no MLflow, persistência em Postgres e uma forma simples de plugar novos domínios." |
| 1:35–2:20 | **A — Ação: arquitetura** | `docs/DOCUMENTACAO.md`, diagrama ou tabela de anéis | "Para resolver isso, separamos a solução em anéis. A API FastAPI ficou responsável pelo produto, autenticação e rotas. O `ml_core_ring` registra domínios, engines e contratos. Os executores concentram o treino pesado. E o Airflow centraliza a orquestração pela DAG `ml_training_dispatch`, que decide o fluxo com base no domínio." |
| 2:20–3:05 | **A — Ação: TC02 ML** | `params.yaml`, `dvc.yaml`, pasta `domains/recommendation` | "No TC02, criamos o domínio `recommendation`. O pipeline DVC roda `preprocess -> feature_eng -> train -> evaluate`. Ele compara Popularity, NMF e um modelo de embeddings em PyTorch. A avaliação usa métricas de ranking como Hit Rate@K, Precision@K, Recall@K, NDCG@K e MAP@K, porque recomendação não é uma classificação simples: importa a ordem dos itens." |
| 3:05–3:55 | **A — Ação: operação** | Swagger: `train/sync`, `promote`, `predict`; Airflow e worker | "Na operação, a API não treina modelos pesados inline. Para recomendação, o treino roda no `worker_recommendation`, acionado diretamente no modo sync de debug ou via Airflow. O run é gravado em `pipeline_runs`, os artefatos e métricas vão para o MLflow, e o promote cria um deployment ativo em `deployed_models`." |
| 3:55–4:35 | **R — Resultado: demo** | Executar ou mostrar resposta de `/predict` | "O resultado é demonstrável pela API. Depois do login, treino rapidamente o `torch_embedding`, promovo o run ativo e chamo `/v1/domains/recommendation/predict` passando `user_id` e `top_k`. A resposta retorna `recommended_items`, além do `pipeline_run_id`, garantindo rastreabilidade entre predição, modelo promovido e treino." |
| 4:35–5:00 | **R — Resultado: valor entregue** | MLflow `tc02_recommendation`, Registry `tc02_recommender`, histórico de deployments | "No final, entregamos uma plataforma extensível: churn e recomendação usam modelos diferentes, mas compartilham autenticação, orquestração, rastreabilidade, promoção, rollback e predição online. O MLflow registra o experimento `tc02_recommendation`, o Registry recebe `tc02_recommender` como efeito colateral, e a fonte de verdade do serving continua no Postgres." |

**Frase de fechamento:** "Em resumo, o projeto saiu de modelos isolados para uma plataforma MLOps multi-domínio, pronta para demonstrar o ciclo completo de Machine Learning em produção: do treino ao consumo por API."

### Easy-run para vídeo curto

Use este fluxo para gravar uma execução simples e demonstrável. Ele privilegia o caminho rápido de TC02 via worker/API; o DVC e o gate completo ficam como validações complementares.

#### 1. Preparar ambiente

```bash
cd /caminho/para/Machine-Learning
cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
make install-dev
python3 scripts/validate_env.py
```

No `.env`, confirmar:

```text
ENVIRONMENT=development
DATABASE_USER=admin
DATABASE_PASS=admin1
DATABASE_NAME=processing
AIRFLOW_DATABASE_NAME=airflow
MLFLOW_DATABASE_NAME=mlflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

#### 2. Subir stack

```bash
make docker-fresh
docker compose ps
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
```

Interfaces:

| Serviço | URL | Login |
|---------|-----|-------|
| Swagger | http://localhost:8000/docs | `admin@admin.com` / `admin1` |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5000 | sem login |
| Dozzle | http://localhost:8888 | sem login |

#### 3. Smoke rápido

```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
curl -s http://localhost:8010/health | python3 -m json.tool
docker exec airflow_scheduler airflow dags list | grep ml_training_dispatch
```

#### 4. Login por terminal

```bash
export API=http://localhost:8000
TOKEN=$(curl -s -X POST "$API/v1/auth/authenticate" \
  -F username=admin@admin.com \
  -F password=admin1 \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
```

#### 5. Treino TC02 rápido

```bash
curl -s -X POST "$API/v1/domains/recommendation/admin/train/sync" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"train_models":["torch_embedding"],"n_epochs":1,"top_k":5,"mlflow_experiment":"tc02_recommendation"}' \
  | python3 -m json.tool
```

Esperado: `status="completed"`, `champion_name="torch_embedding"`, `pipeline_run_id` e métricas de ranking.

#### 6. Promover modelo

```bash
curl -s -X POST "$API/v1/domains/recommendation/admin/promote" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Esperado: `domain="recommendation"`, `status="active"` e, idealmente, `mlflow_registry_model="tc02_recommender"`.

#### 7. Predizer recomendações

```bash
curl -s -X POST "$API/v1/domains/recommendation/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"top_k":5}' \
  | python3 -m json.tool
```

Esperado: `recommended_items` com até 5 IDs e `pipeline_run_id` do modelo ativo.

#### 8. Mostrar evidências finais

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "$API/v1/domains/recommendation/admin/runs?status=completed" \
  | python3 -m json.tool

curl -s -H "Authorization: Bearer $TOKEN" \
  "$API/v1/domains/recommendation/admin/deployments/history" \
  | python3 -m json.tool
```

Abra também:

- MLflow: `http://localhost:5000` → experimento `tc02_recommendation`
- Airflow: `http://localhost:8080` → DAG `ml_training_dispatch`
- Swagger: `http://localhost:8000/docs` → tags `domain-churn` e `domain-recommendation`

#### 9. Validações complementares

Pipeline reprodutível TC02:

```bash
make tc02-repro
cat reports/recommendation/metrics.json | python3 -m json.tool
```

Gate automatizado da plataforma:

```bash
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

Resultado esperado do gate: `PASS: 46+`, `FAIL: 0`.

### Roteiro demo estendido (~15 min)

| # | Acção | Evidência |
|---|-------|-----------|
| 1 | Swagger | Tags `domain-churn`, `domain-recommendation` |
| 2 | `validate_platform.sh` | 0 FAIL |
| 3 | Recommendation train/sync | `pipeline_run_id`, métricas @K |
| 4 | Recommendation promote/predict | `recommended_items` |
| 5 | MLflow | `tc02_recommender` Production |
| 6 | Airflow | DAG `ml_training_dispatch` no histórico |
| 7 | Churn predict | 200 + probabilidade, se já houver deployment ativo |

### Perguntas frequentes

| Pergunta | Resposta |
|----------|----------|
| Por que anéis? | Separar HTTP, ML core, executors e orquestração — extensível por domínio. |
| Como adicionar 3.º domínio? | Plugin + executor + router + branch na DAG dispatch. |
| `/predict` usa Registry? | **Não** — lê `deployed_models` em Postgres. |
| O que foi removido? | `/v1/processor/*`, DAG `ml_training_pipeline`. |

---

## 8. Model Card — Churn (TC01)

| Campo | Valor |
|-------|-------|
| Nome | `churn_classifier` |
| Domínio | `churn` |
| Backend inferência | `mlp` (PyTorch) ou `sklearn` se `USE_MLP_FOR_PREDICTION=false` |
| Dataset | Telco Customer Churn (IBM) |
| Target | `Churn` binário |
| Métrica de negócio | **Recall** (FN >> FP) |
| CV | StratifiedKFold 5-fold, `random_state=42` |

**Limitações:** cohort estático EUA; sem LTV; sem drift online; categorias novas → `handle_unknown="ignore"` (pode degradar silenciosamente).

**Promote:** exactamente 1 run FE `active=true`, `completed`, backend coerente com env. Rotas: `POST /v1/domains/churn/admin/promote`.

**Monitorização sugerida:** PSI semanal; p95 latência < 300ms (`scripts/maintenance/latency_report.py`).

---

## 9. Model Card — Recomendação (TC02)

| Campo | Valor |
|-------|-------|
| Nome Registry | `tc02_recommender` |
| Tipo | User-item embedding PyTorch |
| Dataset | MovieLens ml-latest-small |
| Baselines | Popularity, NMF |
| Métricas | Hit Rate@K, Precision@K, Recall@K, NDCG@K, MAP@K |

**Limitações:** cold-start → popularidade; split temporal por utilizador; MovieLens ≠ catálogo e-commerce real.

**Predict:** `POST /v1/domains/recommendation/predict` com `user_id` + `top_k`.

---

## 10. MLP PyTorch (churn)

| Componente | Valor |
|------------|-------|
| Arquitectura | `Linear(n,64)→ReLU→Dropout→Linear(32)→ReLU→Dropout→Linear(1)` |
| Loss | `BCEWithLogitsLoss` |
| Optimizador | AdamW (lr=1e-3) |
| Early stopping | patience=20 sobre val_loss |
| Pré-processamento | `ColumnTransformer` no bundle `*_preprocess.joblib` |
| Seeds | `torch.manual_seed(42)`, `RANDOM_STATE=42` |

Bundle servido: `.pt` + `_preprocess.joblib` + `_meta.json`.

---

## 11. Observabilidade e manutenção

### Logs HTTP

- Ficheiro: `logs/api_requests/access.jsonl` (JSON por pedido, `duration_ms`)
- Activar: `LOG_HTTP_REQUESTS=true`

### Logs de treino

- Pasta: `data/logs/<timestamp>/pipeline_<ts>.txt`

### Latência agregada (offline)

```bash
python scripts/maintenance/latency_report.py
```

### Drift PSI (offline)

DAG `ml_drift_monitoring` ou:

```bash
python scripts/maintenance/drift_report.py  # treino vs predições exportadas
```

**Nota:** não há alerta de drift em tempo real no `/predict`.

---

## 12. Troubleshooting

| Sintoma | Acção |
|---------|--------|
| Dados MLflow/Airflow velhos | `make docker-fresh` + `rm -rf src/artifacts/mlruns/*` |
| 404 `/domains/…` | `docker compose build api_processing && up -d api_processing` |
| DAG queued | `docker exec airflow_scheduler airflow dags unpause ml_training_dispatch` |
| Promote 400 ambiguidade reco | SQL abaixo ou re-correr validate |
| Trigger 403 | `ENVIRONMENT=development`; reiniciar API |
| pgAdmin não liga | Host `database_processing`, user `admin` |
| Churn promote 400 backend | `USE_MLP_FOR_PREDICTION` alinhado com run FE |
| MLflow crash | `docker compose build mlflow_server && up -d` |

### SQL — desactivar runs reco extra

```bash
source .env
docker exec -e PGPASSWORD="$DATABASE_PASS" database_processing psql \
  -U "$DATABASE_USER" -d "$DATABASE_NAME" -c "
UPDATE pipeline_runs SET active = false
WHERE pipeline_type = 'recommendation'
  AND id != (SELECT MAX(id) FROM pipeline_runs
             WHERE pipeline_type = 'recommendation' AND status = 'completed');
"
```

### Sessão completa copy-paste

```bash
cd /home/gabriel/Machine-Learning
make docker-fresh
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

---

## 13. FAQ

**Equipe:** Alexandre Lucena; Eliane Karasawa; Gabriel Drumond; Marcus de Carvalho; Matheus Pessoa  
**Repo:** https://github.com/Heroi211/Machine-Learning

---

*Última actualização: documentação consolidada — Fase 7 concluída, plataforma só `/v1/domains/*`.*
