PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
POETRY ?= $(shell if command -v poetry >/dev/null 2>&1; then command -v poetry; elif [ -x "$$HOME/.local/bin/poetry" ]; then printf "%s/.local/bin/poetry" "$$HOME"; else printf "poetry"; fi)
POETRY_RUN := $(POETRY) run

# Pacotes em ``src/``; testes importam ``main`` na raiz.
export PYTHONPATH := $(abspath $(CURDIR)/src):$(abspath $(CURDIR))

.PHONY: help install install-dev install-tc02 requirements lock lock-check lint lint-fix format test test-fast coverage run docker-up docker-down docker-fresh clean check check-rings check-rings-strict tc02-repro tc02-validate tc02-promote tc02-up validate-platform validate-platform-infra

help:
	@echo "Alvos principais:"
	@echo "  make install       poetry install --only main"
	@echo "  make install-dev   poetry install --with dev,tc02"
	@echo "  make lock          poetry lock"
	@echo "  make lock-check    poetry check --lock"
	@echo "  make lint          poetry run ruff check"
	@echo "  make lint-fix      poetry run ruff check --fix"
	@echo "  make format        poetry run black em src, tests, main e DAG"
	@echo "  make test          poetry run pytest (cobertura conforme pyproject.toml)"
	@echo "  make test-fast     poetry run pytest sem cobertura (mais rápido)"
	@echo "  make coverage      poetry run pytest com relatórios de cobertura"
	@echo "  make check         lint + test-fast"
	@echo "  make check-rings   docs Fase 0 + avisos de import entre anéis"
	@echo "  make run           poetry run uvicorn local (porta 8000)"
	@echo "  make tc02-repro    poetry run dvc repro"
	@echo "  make docker-up     docker compose up --build"
	@echo "  make docker-down   docker compose down"
	@echo "  make docker-fresh  down -v, prune cache/imagens locais, build --no-cache, up -d"
	@echo "  make validate-platform       validação completa Docker (plataforma, sem churn)"
	@echo "  make validate-platform-infra só infra Docker + health (sem E2E API)"
	@echo "  make clean         artefatos de build e caches locais"

install:
	$(POETRY) install --only main

install-dev:
	$(POETRY) install --with dev,tc02

install-tc02: install-dev

requirements:
	$(PIP) install -r requirements.txt

lock:
	$(POETRY) lock

lock-check:
	$(POETRY) check --lock

lint:
	$(POETRY_RUN) ruff check .

lint-fix:
	$(POETRY_RUN) ruff check --fix .

format:
	$(POETRY_RUN) black src tests main.py airflow/dags

test:
	$(POETRY_RUN) pytest

test-fast:
	$(POETRY_RUN) pytest -q -o addopts=

coverage:
	$(POETRY_RUN) pytest --cov=. --cov-report=html --cov-report=term-missing

run:
	$(POETRY_RUN) uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

# Reset total: contentores, volumes nomeados, imagens locais do compose, cache de build.
# Preserva bind mounts no host (./src, ./data/recommendation, ratings.csv, etc.).
docker-fresh:
	docker compose down -v --remove-orphans --rmi local
	docker builder prune -af
	docker compose build --no-cache --pull
	docker compose up -d --force-recreate
	@echo "Stack limpa e a subir. Acompanhe: docker compose ps"

tc02-repro:
	$(POETRY_RUN) dvc repro

tc02-validate:
	$(POETRY_RUN) python scripts/validate_env.py

tc02-promote:
	$(POETRY_RUN) python scripts/ml/promote_registry.py

tc02-up:
	docker compose up -d --build mlflow_server worker_recommendation

check: lint test-fast

check-rings:
	$(POETRY_RUN) python scripts/check_ring_imports.py

check-rings-strict:
	$(POETRY_RUN) python scripts/check_ring_imports.py --strict

validate-platform:
	@test -n "$$VALIDATE_API_PASSWORD" || (echo "Defina VALIDATE_API_PASSWORD (senha do user seed em init_db/database.sql)"; exit 1)
	bash scripts/validate_platform.sh

validate-platform-infra:
	bash scripts/validate_platform.sh --infra-only --skip-local

clean:
	rm -rf build dist *.egg-info htmlcov .pytest_cache .ruff_cache .coverage coverage.xml
	-find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
