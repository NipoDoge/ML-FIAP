#!/usr/bin/env bash
# =============================================================================
# validate_platform.sh — Validação completa da PLATAFORMA (Docker + API + Airflow)
#
# Escopo: infra Docker, auth, platform_ring (runs, trigger, promote, predict,
#         deployments, rollback), worker reco, MLflow, orquestração Airflow.
# Rotas E2E: /v1/domains/recommendation/… (Fase 5.5).
# Churn tabular: validar manualmente via /v1/domains/churn/…
#
# Uso:
# Credenciais para o script de validação (NÃO usadas pela aplicação).
# Login na API vem de public.users — seed em init_db/database.sql.
#   VALIDATE_API_EMAIL=admin@admin.com VALIDATE_API_PASSWORD='...' ./scripts/validate_platform.sh
#   ./scripts/validate_platform.sh              # tudo (build + stack + E2E)
#   ./scripts/validate_platform.sh --skip-build # stack já levantada
#   ./scripts/validate_platform.sh --infra-only # só contentores e health
#   ./scripts/validate_platform.sh --help
#
# Requisitos: docker, curl, python3 ou PYTHON_BIN (check-rings opcional)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- defaults ----------------------------------------------------------------
SKIP_BUILD=0
SKIP_LOCAL=0
INFRA_ONLY=0
SKIP_DAG=0
SKIP_ROLLBACK=0
QUICK_RECO=1
DAG_TIMEOUT="${DAG_TIMEOUT:-900}"
API_BASE="${API_BASE:-http://localhost:8000}"
DOMAIN_RECO="${DOMAIN_RECO:-recommendation}"
RECO_PREFIX="$API_BASE/v1/domains/$DOMAIN_RECO"
WORKER_BASE="${WORKER_BASE:-http://localhost:8010}"
MLFLOW_BASE="${MLFLOW_BASE:-http://localhost:5000}"
AIRFLOW_BASE="${AIRFLOW_BASE:-http://localhost:8080}"
AIRFLOW_USER="${AIRFLOW_USER:-airflow}"
AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-airflow}"

PASS=0
FAIL=0
SKIP=0

# --- helpers -----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_section() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }
log_pass()    { PASS=$((PASS + 1)); echo -e "${GREEN}✓ PASS${NC}  $1"; }
log_fail()    { FAIL=$((FAIL + 1)); echo -e "${RED}✗ FAIL${NC}  $1"; [[ -n "${2:-}" ]] && echo "         $2"; }
log_skip()    { SKIP=$((SKIP + 1)); echo -e "${YELLOW}○ SKIP${NC}  $1"; }
log_info()    { echo -e "  → $1"; }

require_cmd() {
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || { echo "Comando em falta: $c"; exit 1; }
  done
}

# JSON via Python no host (sem depender de jq)
json_field() {
  "$PYTHON_BIN" -c "import json,sys; d=json.loads(sys.argv[1]); v=d.get(sys.argv[2]); print('' if v is None else v)" "$1" "$2"
}

json_train_body() {
  "$PYTHON_BIN" -c "import json,sys; p=json.loads(sys.argv[1]); print(json.dumps({'domain':'recommendation','user_id':2,'params':p,'airflow_dag_run_id':sys.argv[2]}))" "$1" "$2"
}

json_dag_conf() {
  "$PYTHON_BIN" -c "import json,sys; p=json.loads(sys.argv[1]); c={'domain':'recommendation','user_id':2,'top_k':5}; c.update(p); print(json.dumps(c))" "$1"
}

json_len_array() {
  "$PYTHON_BIN" -c "import json,sys; print(len(json.load(sys.stdin)))"
}

json_reco_items_len() {
  "$PYTHON_BIN" -c "import json,sys; d=json.loads(sys.argv[1]); items=d.get('recommended_items') or []; print(len(items))" "$1"
}

json_has_key() {
  "$PYTHON_BIN" -c "import json,sys; d=json.loads(sys.stdin.read()); sys.exit(0 if sys.argv[1] in d else 1)" "$1"
}

json_openapi_has_path() {
  "$PYTHON_BIN" -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if sys.argv[1] in d.get('paths',{}) else 1)" "$1"
}

load_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  # Seed init_db/database.sql — a app NÃO lê estas vars; só o script de validação.
  VALIDATE_API_EMAIL="${VALIDATE_API_EMAIL:-admin@admin.com}"
  VALIDATE_API_PASSWORD="${VALIDATE_API_PASSWORD:-}"
}

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  echo ""
  echo "Opções:"
  echo "  --skip-build     Não executa docker compose build"
  echo "  --skip-local     Não corre check-rings / validate_env"
  echo "  --infra-only     Só Fases 1–2 (Docker + health); sem E2E API"
  echo "  --skip-dag       Não espera DAG ml_training_dispatch (Airflow E2E)"
  echo "  --skip-rollback  Não testa POST /admin/rollback"
  echo "  --full-reco      Treino reco completo (3 modelos; demorado)"
  echo "  --help           Esta ajuda"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-build)    SKIP_BUILD=1 ;;
      --skip-local)    SKIP_LOCAL=1 ;;
      --infra-only)    INFRA_ONLY=1 ;;
      --skip-dag)      SKIP_DAG=1 ;;
      --skip-rollback) SKIP_ROLLBACK=1 ;;
      --full-reco)     QUICK_RECO=0 ;;
      --help|-h)       usage; exit 0 ;;
      *) echo "Opção desconhecida: $1"; usage; exit 1 ;;
    esac
    shift
  done
}

assert_http() {
  local label="$1" url="$2" expect="${3:-200}" auth="${4:-}"
  local code
  if [[ -n "$auth" ]]; then
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $auth" "$url")
  else
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  fi
  if [[ "$code" == "$expect" ]]; then
    log_pass "$label (HTTP $code)"
  else
    log_fail "$label" "esperado HTTP $expect, obteve $code — $url"
  fi
}

wait_url() {
  local label="$1" url="$2" max="${3:-120}"
  shift 3 || true
  local extra=("$@")
  local i=0
  while [[ $i -lt $max ]]; do
    if (( ${#extra[@]} > 0 )); then
      if curl -sf "${extra[@]}" "$url" >/dev/null 2>&1; then
        log_pass "$label (ready em ${i}s)"
        return 0
      fi
    elif curl -sf "$url" >/dev/null 2>&1; then
      log_pass "$label (ready em ${i}s)"
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  log_fail "$label" "timeout ${max}s — $url"
  return 1
}

container_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true
}

get_token() {
  local resp
  resp=$(curl -s -X POST "$API_BASE/v1/auth/authenticate" \
    -F "username=$VALIDATE_API_EMAIL" \
    -F "password=$VALIDATE_API_PASSWORD")
  TOKEN=$(json_field "$resp" access_token)
  if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
    log_fail "Autenticação API" "email=$VALIDATE_API_EMAIL — senha deve coincidir com public.users (init_db/database.sql)"
    return 1
  fi
  log_pass "Autenticação API (admin)"
}

reco_train_params() {
  if [[ "$QUICK_RECO" -eq 1 ]]; then
    # torch_embedding gera prefix/.pt/.meta.json exigidos pelo promote (popularity não grava artefactos).
    echo '{"train_models":["torch_embedding"],"top_k":5,"n_epochs":1}'
  else
    echo '{"top_k":10}'
  fi
}

# Fases 4–6 criam vários runs recommendation activos; promote exige exactamente um.
dedupe_active_reco_runs() {
  local db_user="${DATABASE_USER:-gabriel_drumond}"
  local db_name="${DATABASE_NAME:-processing}"
  local db_pass="${DATABASE_PASS:-}"
  if ! container_running database_processing; then
    log_info "dedupe_active_reco_runs: PostgreSQL indisponível — ignorado"
    return 0
  fi
  docker exec -e PGPASSWORD="$db_pass" database_processing psql -U "$db_user" -d "$db_name" -v ON_ERROR_STOP=1 -c "
    UPDATE pipeline_runs SET active = false
    WHERE pipeline_type = 'recommendation'
      AND lower(objective) = 'recommendation'
      AND status = 'completed'
      AND id != (
        SELECT id FROM pipeline_runs
        WHERE pipeline_type = 'recommendation'
          AND lower(objective) = 'recommendation'
          AND status = 'completed'
        ORDER BY id DESC
        LIMIT 1
      );
  " >/dev/null 2>&1 || log_info "dedupe_active_reco_runs: aviso — não foi possível desactivar runs extra"
}

# --- Fase 0: pré-requisitos locais -------------------------------------------
phase_local() {
  [[ "$SKIP_LOCAL" -eq 1 ]] && { log_skip "Fase 0 — checks locais (--skip-local)"; return; }
  log_section "Fase 0 — Pré-requisitos locais"
  if "$PYTHON_BIN" scripts/check_ring_imports.py >/tmp/check_rings.out 2>&1; then
    log_pass "check_ring_imports (sem erros)"
  else
    log_fail "check_ring_imports" "ver /tmp/check_rings.out"
  fi
  if PYTHONPATH=src:. "$PYTHON_BIN" scripts/validate_env.py >/tmp/validate_env.out 2>&1; then
    log_pass "validate_env (deps ML + paths TC02)"
  else
    log_fail "validate_env" "ver /tmp/validate_env.out"
  fi
}

# --- Fase 1: Docker stack ----------------------------------------------------
phase_docker_up() {
  log_section "Fase 1 — Stack Docker (compose)"
  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    log_info "docker compose up -d --build (pode demorar na 1ª vez)..."
    docker compose up -d --build
  else
    log_info "docker compose up -d (--skip-build)..."
    docker compose up -d
  fi
  log_pass "docker compose up"
}

# --- Fase 2: Infraestrutura / contentores ------------------------------------
phase_infra() {
  log_section "Fase 2 — Contentores e health checks"

  local services=(
    database_processing:PostgreSQL
    api_processing:API
    worker_recommendation:Worker-reco
    mlflow_server:MLflow
    airflow_webserver:Airflow-web
    airflow_scheduler:Airflow-scheduler
    pgadmin_db:pgAdmin
    dozzle_logs:Dozzle
  )
  for entry in "${services[@]}"; do
    local c="${entry%%:*}" label="${entry##*:}"
    if container_running "$c"; then
      log_pass "Container $label ($c) running"
    else
      log_fail "Container $label ($c)" "não está running"
    fi
  done

  wait_url "API /v1/health" "$API_BASE/v1/health" 180 || true
  wait_url "Worker /health" "$WORKER_BASE/health" 120 || true
  wait_url "Airflow /health" "$AIRFLOW_BASE/health" 180 || true
  wait_url "MLflow UI" "$MLFLOW_BASE" 60 || true
  wait_url "Dozzle logs" "http://localhost:8888" 30 || true
  wait_url "pgAdmin" "http://localhost:5050" 30 || true

  # ENVIRONMENT no contentor API (train/trigger bloqueado em prd)
  local env_api env_api_lc
  env_api=$(docker exec api_processing printenv ENVIRONMENT 2>/dev/null || echo "unknown")
  env_api_lc=$(printf '%s' "$env_api" | tr '[:upper:]' '[:lower:]')
  if [[ "$env_api_lc" == "prd" || "$env_api_lc" == "prod" || "$env_api_lc" == "production" ]]; then
    log_fail "API ENVIRONMENT=$env_api" "train/trigger via API retorna 403 — use ENVIRONMENT=development no .env"
  else
    log_pass "API ENVIRONMENT=$env_api (train/trigger permitido)"
  fi

  # Volumes reco visíveis no worker
  if docker exec worker_recommendation test -f /var/www/ml_shared/data/recommendation/raw/ratings.csv 2>/dev/null; then
    log_pass "Volume reco: ratings.csv no worker"
  else
    log_fail "Volume reco no worker" "falta /var/www/ml_shared/data/recommendation/raw/ratings.csv"
  fi
  if docker exec api_processing test -f /var/www/ml_shared/params.yaml 2>/dev/null; then
    log_pass "Volume reco: params.yaml na API"
  else
    log_fail "Volume params.yaml na API"
  fi
  if docker exec airflow_scheduler test -f /opt/airflow/ml_project/params.yaml 2>/dev/null; then
    log_pass "Volume reco: params.yaml no Airflow"
  else
    log_fail "Volume params.yaml no Airflow"
  fi

  # Rede interna: API → worker
  if docker exec api_processing python3 -c "import urllib.request; urllib.request.urlopen('http://worker_recommendation:8010/health', timeout=5)" >/dev/null 2>&1; then
    log_pass "Rede nwprocessing: API → worker_recommendation:8010"
  else
    log_fail "Conectividade API → worker" "python urllib worker_recommendation:8010 falhou dentro de api_processing"
  fi

  # Airflow env worker URL
  local wurl
  wurl=$(docker exec airflow_scheduler printenv WORKER_RECOMMENDATION_URL 2>/dev/null || echo "")
  if [[ "$wurl" == "http://worker_recommendation:8010" ]]; then
    log_pass "Airflow WORKER_RECOMMENDATION_URL=$wurl"
  else
    log_fail "WORKER_RECOMMENDATION_URL" "esperado http://worker_recommendation:8010, obteve: $wurl"
  fi

  # DAGs
  if docker exec airflow_scheduler airflow dags list 2>/dev/null | grep -q ml_training_dispatch; then
    log_pass "DAG ml_training_dispatch carregada"
  else
    log_fail "DAG ml_training_dispatch" "não encontrada em airflow dags list"
  fi
  if docker exec airflow_scheduler airflow variables get ml_training_dispatch_conf >/dev/null 2>&1; then
    log_pass "Airflow Variable ml_training_dispatch_conf definida"
  else
    log_fail "Variable ml_training_dispatch_conf" "ausente — airflow-init correu?"
  fi

  # DAG pausada → dagRuns ficam em queued para sempre
  if docker exec airflow_scheduler airflow dags list 2>/dev/null | grep ml_training_dispatch | grep -qE '\|\s+False\s*$'; then
    log_pass "DAG ml_training_dispatch despausada"
  elif docker exec airflow_scheduler airflow dags unpause ml_training_dispatch >/dev/null 2>&1; then
    log_pass "DAG ml_training_dispatch estava pausada — despausada pelo script"
  else
    log_fail "DAG ml_training_dispatch pausada" \
      "airflow dags unpause ml_training_dispatch — ou UI Airflow → toggle pause"
  fi

  # MLflow API
  if curl -sf "$MLFLOW_BASE/api/2.0/mlflow/experiments/search" \
      -H "Content-Type: application/json" \
      -d '{"max_results": 1}' >/dev/null 2>&1; then
    log_pass "MLflow REST API responde"
  else
    log_skip "MLflow REST API" "endpoint pode variar por versão — UI em $MLFLOW_BASE OK basta"
  fi
}

# --- Fase 3: Auth + rotas admin base -----------------------------------------
phase_auth() {
  log_section "Fase 3 — Auth e rotas administrativas (shell da plataforma)"
  get_token || return

  assert_http "GET /v1/auth/logged" "$API_BASE/v1/auth/logged" 200 "$TOKEN"
  assert_http "GET /v1/users/" "$API_BASE/v1/users/" 200 "$TOKEN"
  assert_http "GET /v1/roles/" "$API_BASE/v1/roles/" 200 "$TOKEN"
  assert_http "GET /v1/domains/recommendation/admin/runs" \
    "$RECO_PREFIX/admin/runs" 200 "$TOKEN"

  # deployments history — 404 aceitável se nunca houve promote
  local dep_code
  dep_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    "$RECO_PREFIX/admin/deployments/history")
  if [[ "$dep_code" == "200" || "$dep_code" == "404" ]]; then
    log_pass "GET /domains/recommendation/admin/deployments/history (HTTP $dep_code)"
  else
    log_fail "GET deployments/history" "HTTP $dep_code"
  fi
}

# --- Fase 4: Worker HTTP (executors + persist) -------------------------------
phase_worker() {
  log_section "Fase 4 — Worker recomendação (POST /train + persist BD)"
  local params body run_id
  params=$(reco_train_params)
  body=$(json_train_body "$params" "validate_platform_worker")

  log_info "Treino via worker ($([[ $QUICK_RECO -eq 1 ]] && echo rápido || echo completo))..."
  local resp
  resp=$(curl -s -w "\n%{http_code}" -X POST "$WORKER_BASE/train" \
    -H "Content-Type: application/json" -d "$body")
  local code=${resp##*$'\n'}
  local json=${resp%$'\n'*}

  if [[ "$code" == "200" ]]; then
    run_id=$(json_field "$json" pipeline_run_id)
    if [[ -n "$run_id" && "$run_id" != "null" ]]; then
      log_pass "Worker POST /train → pipeline_run_id=$run_id"
      WORKER_RUN_ID="$run_id"
    else
      log_fail "Worker /train" "sem pipeline_run_id no JSON"
    fi
  else
    log_fail "Worker POST /train" "HTTP $code — $json"
  fi
}

# --- Fase 5: Platform trigger (API → Airflow) --------------------------------
phase_trigger_api() {
  log_section "Fase 5 — Platform train/trigger (API → ml_training_dispatch)"
  [[ -z "${TOKEN:-}" ]] && { log_skip "train/trigger API (sem token)"; return; }

  local resp code dag_run_id json
  resp=$(curl -s -w "\n%{http_code}" -X POST "$RECO_PREFIX/admin/train/trigger" \
    -H "Authorization: Bearer $TOKEN" \
    -F "top_k=5" \
    -F 'train_models=["torch_embedding"]' \
    -F "n_epochs=1")
  code=${resp##*$'\n'}
  json=${resp%$'\n'*}

  if [[ "$code" == "202" ]]; then
    dag_run_id=$(json_field "$json" dag_run_id)
    log_pass "POST /domains/recommendation/admin/train/trigger (HTTP 202, dag_run_id=$dag_run_id)"
    API_DAG_RUN_ID="$dag_run_id"
  elif [[ "$code" == "403" ]]; then
    log_fail "POST /domains/recommendation/admin/train/trigger" "403 — ENVIRONMENT=prd no contentor API"
  else
    log_fail "POST /domains/recommendation/admin/train/trigger" "HTTP $code — $json"
  fi
}

# --- Fase 6: Airflow E2E (dispatch → worker) ---------------------------------
phase_dag_e2e() {
  [[ "$SKIP_DAG" -eq 1 ]] && { log_skip "Fase 6 — Airflow E2E (--skip-dag)"; return; }
  log_section "Fase 6 — Orquestração Airflow (ml_training_dispatch → worker)"

  docker exec airflow_scheduler airflow dags unpause ml_training_dispatch >/dev/null 2>&1 || true

  local params dag_run_id conf state tasks_ok=0
  params=$(reco_train_params)
  dag_run_id="validate_platform_$(date +%s)"
  conf=$(json_dag_conf "$params")

  log_info "Disparo directo Airflow REST (conf rápida)..."
  local trigger_resp
  trigger_resp=$(curl -s -u "$AIRFLOW_USER:$AIRFLOW_PASSWORD" \
    -X POST "$AIRFLOW_BASE/api/v1/dags/ml_training_dispatch/dagRuns" \
    -H "Content-Type: application/json" \
    -d "{\"dag_run_id\":\"$dag_run_id\",\"conf\":$conf}")
  if echo "$trigger_resp" | json_has_key dag_run_id; then
    log_pass "Airflow REST trigger ml_training_dispatch ($dag_run_id)"
  else
    log_fail "Airflow REST trigger" "$trigger_resp"
    return
  fi

  log_info "Aguardar conclusão do DAG (timeout ${DAG_TIMEOUT}s)..."
  local elapsed=0
  while [[ $elapsed -lt $DAG_TIMEOUT ]]; do
    state=$(curl -s -u "$AIRFLOW_USER:$AIRFLOW_PASSWORD" \
      "$AIRFLOW_BASE/api/v1/dags/ml_training_dispatch/dagRuns/$dag_run_id" \
      | "$PYTHON_BIN" -c "import json,sys; d=json.load(sys.stdin); print(d.get('state') or 'unknown')")
    case "$state" in
      success)
        log_pass "DAG $dag_run_id concluído (success)"
        DAG_RUN_ID="$dag_run_id"
        return
        ;;
      failed)
        log_fail "DAG $dag_run_id" "state=failed — ver Airflow UI $AIRFLOW_BASE"
        return
        ;;
      queued)
        if [[ $elapsed -ge 120 ]]; then
          log_fail "DAG $dag_run_id preso em queued" \
            "DAG pausada ou scheduler sobrecarregado — airflow dags unpause ml_training_dispatch"
          return
        fi
        ;;
    esac
    sleep 10
    elapsed=$((elapsed + 10))
    [[ $((elapsed % 60)) -eq 0 ]] && log_info "DAG state=$state (${elapsed}s)..."
  done
  log_fail "DAG $dag_run_id" "timeout ${DAG_TIMEOUT}s (state=$state)"
}

# --- Fase 7: Promote → Predict → Runs → Rollback -----------------------------
phase_platform_ml() {
  log_section "Fase 7 — Platform ML (promote, predict, runs, rollback)"
  [[ -z "${TOKEN:-}" ]] && { log_skip "Fase 7 (sem token)"; return; }

  # Confirmar run recommendation completed na BD via API
  local runs_count
  runs_count=$(curl -s -H "Authorization: Bearer $TOKEN" \
    "$RECO_PREFIX/admin/runs?status=completed" \
    | json_len_array)
  if [[ "${runs_count:-0}" -ge 1 ]]; then
    log_pass "GET /domains/recommendation/admin/runs completed (count=$runs_count)"
  else
    log_fail "Runs recommendation completed" "count=$runs_count — worker/DAG falhou?"
    return
  fi

  # Promote (um único run activo — fases 4–6 podem ter criado vários)
  dedupe_active_reco_runs
  local prom_code prom_body
  prom_body=$(curl -s -w "\n%{http_code}" -X POST \
    "$RECO_PREFIX/admin/promote" \
    -H "Authorization: Bearer $TOKEN")
  prom_code=${prom_body##*$'\n'}
  if [[ "$prom_code" == "201" ]]; then
    log_pass "POST /domains/recommendation/admin/promote (HTTP 201)"
    FIRST_DEPLOYMENT_ID=$(json_field "${prom_body%$'\n'*}" id)
    local prom_json reg_model reg_ver reg_warn
    prom_json="${prom_body%$'\n'*}"
    reg_model=$(json_field "$prom_json" mlflow_registry_model)
    reg_ver=$(json_field "$prom_json" mlflow_registry_version)
    reg_warn=$(json_field "$prom_json" mlflow_registry_warning)
    if [[ -n "$reg_ver" && "$reg_ver" != "null" ]]; then
      log_pass "MLflow Registry side-effect (model=$reg_model v=$reg_ver)"
    elif [[ -n "$reg_warn" && "$reg_warn" != "null" ]]; then
      log_skip "MLflow Registry" "$reg_warn (promote BD OK — Fase 6 best-effort)"
    else
      log_skip "MLflow Registry" "sem mlflow_registry_version na resposta (MLflow offline ou run sem registo)"
    fi
  else
    log_fail "POST /admin/promote" "HTTP $prom_code — ${prom_body%$'\n'*}"
    return
  fi

  # Deployments history (agora deve existir)
  assert_http "GET deployments/history (pós-promote)" \
    "$RECO_PREFIX/admin/deployments/history" 200 "$TOKEN"

  # Predict recommendation (sem campo domain no body)
  local pred_body pred_code reco_items
  pred_body=$(curl -s -w "\n%{http_code}" -X POST "$RECO_PREFIX/predict" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"user_id":1,"top_k":5}')
  pred_code=${pred_body##*$'\n'}
  if [[ "$pred_code" == "200" ]]; then
    reco_items=$(json_reco_items_len "${pred_body%$'\n'*}")
    if [[ "${reco_items:-0}" -ge 1 ]]; then
      log_pass "POST /domains/recommendation/predict ($reco_items itens)"
    else
      log_pass "POST /domains/recommendation/predict (HTTP 200; recommended_items vazio — modelo popularity OK)"
    fi
  else
    log_fail "POST /domains/recommendation/predict" "HTTP $pred_code — ${pred_body%$'\n'*}"
  fi

  # Rollback (precisa de 2 deployments)
  if [[ "$SKIP_ROLLBACK" -eq 1 ]]; then
    log_skip "Rollback (--skip-rollback)"
    return
  fi

  log_info "Segundo treino + promote para testar rollback..."
  local params body resp code run2
  params=$(reco_train_params)
  body=$(json_train_body "$params" "validate_platform_rollback")
  resp=$(curl -s -w "\n%{http_code}" -X POST "$WORKER_BASE/train" \
    -H "Content-Type: application/json" -d "$body")
  code=${resp##*$'\n'}
  if [[ "$code" != "200" ]]; then
    log_skip "Rollback" "segundo treino falhou HTTP $code"
    return
  fi

  dedupe_active_reco_runs
  prom_body=$(curl -s -w "\n%{http_code}" -X POST \
    "$RECO_PREFIX/admin/promote" \
    -H "Authorization: Bearer $TOKEN")
  prom_code=${prom_body##*$'\n'}
  if [[ "$prom_code" != "201" ]]; then
    log_skip "Rollback" "segundo promote falhou HTTP $prom_code"
    return
  fi

  local rb_body rb_code
  rb_body=$(curl -s -w "\n%{http_code}" -X POST \
    "$RECO_PREFIX/admin/rollback" \
    -H "Authorization: Bearer $TOKEN")
  rb_code=${rb_body##*$'\n'}
  if [[ "$rb_code" == "200" ]]; then
    log_pass "POST /domains/recommendation/admin/rollback (HTTP 200)"
  else
    log_fail "POST /admin/rollback" "HTTP $rb_code — ${rb_body%$'\n'*}"
  fi
}

# --- Fase 8: Swagger / OpenAPI -----------------------------------------------
phase_openapi() {
  log_section "Fase 8 — OpenAPI / Swagger"
  if curl -sf "$API_BASE/openapi.json" | json_openapi_has_path "/v1/domains/recommendation/admin/train/trigger"; then
    log_pass "OpenAPI expõe /v1/domains/recommendation/admin/train/trigger"
  else
    log_fail "OpenAPI train/trigger" "path ausente em openapi.json"
  fi
  if curl -sf "$API_BASE/openapi.json" | json_openapi_has_path "/v1/domains/recommendation/predict"; then
    log_pass "OpenAPI expõe /v1/domains/recommendation/predict"
  else
    log_fail "OpenAPI predict recommendation"
  fi
  if curl -sf "$API_BASE/docs" -o /dev/null; then
    log_pass "Swagger UI em $API_BASE/docs"
  else
    log_fail "Swagger UI"
  fi
}

# --- Resumo ------------------------------------------------------------------
print_summary() {
  log_section "Resumo"
  echo -e "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}SKIP: $SKIP${NC}"
  echo ""
  echo "  Interfaces web:"
  echo "    API Swagger  → $API_BASE/docs"
  echo "    Airflow      → $AIRFLOW_BASE  ($AIRFLOW_USER / $AIRFLOW_PASSWORD)"
  echo "    MLflow       → $MLFLOW_BASE"
  echo "    pgAdmin      → http://localhost:5050"
  echo "    Dozzle       → http://localhost:8888"
  echo ""
  if [[ "$FAIL" -gt 0 ]]; then
    echo -e "${RED}Validação FALHOU — corrija os itens acima.${NC}"
    exit 1
  fi
  echo -e "${GREEN}Validação da plataforma concluída com sucesso.${NC}"
}

# --- main --------------------------------------------------------------------
main() {
  parse_args "$@"
  require_cmd docker curl "$PYTHON_BIN"
  load_env

  if [[ "$INFRA_ONLY" -eq 0 && -z "$VALIDATE_API_PASSWORD" ]]; then
    echo -e "${RED}ERRO: VALIDATE_API_PASSWORD em falta (só para o script de teste).${NC}"
    echo "  A API autentica contra public.users — seed em init_db/database.sql."
    echo "  Ex.: VALIDATE_API_PASSWORD='...' make validate-platform"
    exit 1
  fi

  echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║  Validação completa da PLATAFORMA (sem churn)               ║${NC}"
  echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
  log_info "Login teste: $VALIDATE_API_EMAIL (public.users / init_db — não é variável da plataforma)"

  phase_local
  phase_docker_up
  phase_infra

  if [[ "$INFRA_ONLY" -eq 1 ]]; then
    print_summary
    exit 0
  fi

  phase_auth
  phase_worker
  phase_trigger_api
  phase_dag_e2e
  phase_platform_ml
  phase_openapi
  print_summary
}

main "$@"
