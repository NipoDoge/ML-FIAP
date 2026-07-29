import glob
import logging
import math
import os
import json
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.custom_logger import setup_pipeline_run_logging
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from platform_ring.schemas.contracts import InferenceReport
from services.processor.fe_bundle_export import (
    finalize_fe_bundle_pipeline_outputs,
    prepare_fe_bundle_baseline_tree,
    write_fe_manifest_zip_from_run_root,
)
from services.processor.inference_report import (
    attach_fe_model_comparison_table,
    attach_mlp_metrics_snapshot,
    build_inference_report,
    build_recommendation_inference_report,
)
from services.utils import utcnow

logger = logging.getLogger(__name__)


def _resolve_snapshot_manifest_and_sample(snap_root: str) -> tuple[str, str]:
    """
    Localiza ``manifest*.json`` e ``baseline_sample*.csv`` na pasta de snapshot
    (nomes sem sufixo ou com sufixo tipo ``_automatic``).
    """
    root = os.path.abspath(snap_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Snapshot inexistente: {root}")
    manifests = glob.glob(os.path.join(root, "manifest*.json"))
    if not manifests:
        raise FileNotFoundError(f"Nenhum manifest*.json em {root}")
    manifests.sort(key=lambda p: (0 if os.path.basename(p) == "manifest.json" else 1, p))
    samples = glob.glob(os.path.join(root, "baseline_sample*.csv"))
    if not samples:
        raise FileNotFoundError(f"Nenhum baseline_sample*.csv em {root}")
    samples.sort(key=lambda p: (0 if os.path.basename(p) == "baseline_sample.csv" else 1, p))
    return manifests[0], samples[0]


async def fetch_active_baseline_metrics_snapshot(session: AsyncSession, objective: str) -> dict | None:
    """
    Lê o baseline ``completed`` + ``active`` para o domínio — usado ao fechar o FE para o relatório
    (baseline sklearn vs MLP / FE na mesma resposta de ``/predict``).
    """
    d = objective.strip().lower()
    stmt = (
        select(PipelineRuns)
        .where(
            PipelineRuns.pipeline_type == "baseline",
            PipelineRuns.active.is_(True),
            PipelineRuns.status == "completed",
            func.lower(PipelineRuns.objective) == d,
        )
        .limit(1)
    )
    res = await session.execute(stmt)
    row = res.scalars().one_or_none()
    if row is None:
        return None
    m = dict(row.metrics or {})

    def _f(*keys: str) -> float | None:
        for key in keys:
            raw = m.get(key)
            if raw is None:
                continue
            try:
                x = float(raw)
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                continue
        return None

    return {
        "baseline_pipeline_run_id": row.id,
        "role": "baseline_sklearn",
        "description": (
            "Métricas de **teste** do pipeline Baseline (sklearn, p.ex. regressão logística + "
            "pré-processamento do contrato). Referência **antes** do feature engineering; não é o modelo servido em /predict."
        ),
        "test_accuracy": _f("test_accuracy"),
        "test_precision": _f("test_precision"),
        "test_recall": _f("test_recall"),
        "test_f1": _f("test_f1"),
        "test_pr_auc": _f("test_pr_auc"),
        "test_roc_auc": _f("test_roc_auc"),
        "classification_decision_threshold": _f("classification_decision_threshold"),
    }


def _recall_from_metrics(metrics: dict | None) -> float:
    try:
        return float((metrics or {}).get("test_recall", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _fe_competition_score_from_metrics(metrics: dict | None) -> float:
    """
    Score usado no desempate entre runs FE: **validação cruzada** da métrica de optimização
    (ex. ``cv_recall`` quando ``optimization_metric`` é ``recall``), não o valor no conjunto
    de teste — alinhado ao critério do tuning.
    """
    if not metrics:
        return float("-inf")
    opt = (metrics.get("optimization_metric") or "").strip().lower()
    if not opt:
        return float("-inf")
    for key in (f"cv_{opt}", "best_cv_score"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return float("-inf")


def _fe_model_metric_pair(metrics: dict | None) -> tuple[str, str]:
    m = metrics or {}
    name = (m.get("best_model_name") or "").strip()
    opt = (m.get("optimization_metric") or "").strip().lower()
    return name, opt


async def _baseline_recall_winner(session: AsyncSession, run: PipelineRuns, run_timestamp: str) -> None:
    """
    Mantém no máximo um baseline ``active`` por ``objective``: compara ``test_recall`` (teste)
    com outros baselines ``completed`` e ``active``; se o novo for estritamente melhor,
    desactiva os anteriores, **publica** ``pre_processed/*`` partir do snapshot, e marca o novo ativo;
    caso contrário desactiva o novo (**sem** tocar no contrato FE global).

    ``run_timestamp``: pasta ``PATH_DATA/PATH_LOGS/<ts>`` com manifest/sample da corrida.
    """
    if run.pipeline_type != "baseline" or run.status != "completed":
        return

    new_recall = _recall_from_metrics(run.metrics)
    obj = (run.objective or "").strip().lower()

    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != run.id,
            PipelineRuns.pipeline_type == "baseline",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            PipelineRuns.is_airflow_run.is_(run.is_airflow_run),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    champions = list(res.scalars().all())

    if not champions:
        _publish_global_baseline_from_snapshot(run_timestamp)
        run.active = True
        logger.info(
            "Baseline run %s primeiro baseline ou sem campões recall — manifest global publicado.",
            run.id,
        )
        return

    best_prev = max((_recall_from_metrics(c.metrics) for c in champions), default=float("-inf"))

    if new_recall > best_prev:
        _publish_global_baseline_from_snapshot(run_timestamp)
        for c in champions:
            c.active = False
            session.add(c)
            logger.info(
                "Baseline run %s desactivado pelo comparador recall (mantido run %s, test_recall=%.6f > %.6f).",
                c.id,
                run.id,
                new_recall,
                best_prev,
            )
        run.active = True
    else:
        run.active = False
        logger.info(
            "Baseline run %s desactivado: test_recall=%.6f <= melhor anterior=%.6f.",
            run.id,
            new_recall,
            best_prev,
        )


async def _deactivate_other_fe_runs_for_objective(
    session: AsyncSession, *, objective: str, keep_run_id: int
) -> None:
    """Garante no máximo um FE `active` por domínio: desliga todos os outros concluídos."""
    obj = objective.strip().lower()
    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != keep_run_id,
            PipelineRuns.pipeline_type == "feature_engineering",
            PipelineRuns.active.is_(True),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    for other in res.scalars().all():
        other.active = False
        session.add(other)
        logger.info(
            "FE run %s desactivado (só um campeão activo por objective): cede lugar ao run %s.",
            other.id,
            keep_run_id,
        )


async def _fe_recall_winner(session: AsyncSession, run: PipelineRuns) -> None:
    """
    Mantém no máximo um FE ``active`` por ``objective`` quando há linhagem comparável:
    compara o **melhor score em CV** da ``optimization_metric`` (ex. ``cv_recall``), não métricas
    no holdout de teste; exige ``best_model_name`` e ``optimization_metric`` iguais a um campeão.
    Não altera ``pre_processed`` (contrato continua a ser do baseline).

    Após marcar um run como vencedor, **desactiva qualquer outro** FE activo do mesmo objective
    (independentemente de Airflow vs manual ou ``inference_backend``), para evitar dois ``active``
    simultâneos e confusão no promote/predict.
    """
    if run.pipeline_type != "feature_engineering" or run.status != "completed":
        return

    new_score = _fe_competition_score_from_metrics(run.metrics)
    new_name, new_metric = _fe_model_metric_pair(run.metrics)
    obj = (run.objective or "").strip().lower()

    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != run.id,
            PipelineRuns.pipeline_type == "feature_engineering",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            PipelineRuns.is_airflow_run.is_(run.is_airflow_run),
            PipelineRuns.inference_backend == (run.inference_backend or "sklearn"),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    champions = list(res.scalars().all())

    if not champions:
        await _deactivate_other_fe_runs_for_objective(session, objective=run.objective, keep_run_id=run.id)
        run.active = True
        logger.info(
            "FE run %s primeiro FE ou sem campeões activos no objective %r — marcado activo.",
            run.id,
            obj,
        )
        return

    compatible = [c for c in champions if _fe_model_metric_pair(c.metrics) == (new_name, new_metric)]
    if not compatible:
        run.active = False
        champ_summary = [
            {
                "id": c.id,
                "best_model_name": _fe_model_metric_pair(c.metrics)[0],
                "optimization_metric": _fe_model_metric_pair(c.metrics)[1],
            }
            for c in champions
        ]
        logger.warning(
            "FE run %s desactivado: best_model_name=%r ou optimization_metric=%r não coincidem com "
            "campeões activos %s. Requer validação manual antes de promover.",
            run.id,
            new_name,
            new_metric,
            champ_summary,
        )
        return

    best_prev = max((_fe_competition_score_from_metrics(c.metrics) for c in compatible), default=float("-inf"))

    if new_score > best_prev:
        await _deactivate_other_fe_runs_for_objective(session, objective=run.objective, keep_run_id=run.id)
        run.active = True
        logger.info(
            "FE run %s campeão cv_%s (score=%.6f > %.6f, modelo=%r métrica_optim=%r). Outros FE do objective desactivados.",
            run.id,
            new_metric,
            new_score,
            best_prev,
            new_name,
            new_metric,
        )
    else:
        run.active = False
        logger.info(
            "FE run %s desactivado: cv_%s=%.6f <= melhor anterior=%.6f (modelo=%r métrica_optim=%r).",
            run.id,
            new_metric,
            new_score,
            best_prev,
            new_name,
            new_metric,
        )


def _sklearn_feature_names(model) -> list[str] | None:
    """Nomes de features esperados pelo estimador sklearn (Pipeline ou estimador único)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def _align_dataframe_to_model(model, df: pd.DataFrame) -> pd.DataFrame:
    names = _sklearn_feature_names(model)
    if not names:
        return df
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas em falta para predição (alinhar ao treino): " + ", ".join(missing)
        )
    return df[names]


def _baseline_clean_encode_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alinha uma linha ao mesmo esquema de features do treino do Baseline.
    Imputação e one-hot ficam no ``Pipeline`` persistido (``model.predict``).
    """
    df_clean = df.copy()
    if "dataset" in df_clean.columns:
        df_clean = df_clean.drop(columns=["dataset"])
    if "target" in df_clean.columns:
        df_clean = df_clean.drop(columns=["target"])
    for col in list(df_clean.columns):
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(np.int8)
    return df_clean


_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY = "strategy_monthly_charges_median"


def _hydrate_strategy_from_run_metrics(strategy, run: PipelineRuns) -> None:
    """Recarrega estado da strategy guardado em ``run.metrics`` (ex.: mediana no Churn)."""
    m = run.metrics
    if not m:
        return
    if _STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY in m and hasattr(strategy, "monthly_median"):
        try:
            strategy.monthly_median = float(m[_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY])
        except (TypeError, ValueError):
            pass


def _infer_train_matrix_payload(features: dict) -> None:
    """Levanta erro claro se o cliente enviar colunas típicas de ``train_model_input`` (pós-OHE / pós-escala)."""
    keys = [str(k).strip().lower() for k in features]
    ohe_like = (
        "internetservice_",
        "contract_",
        "paymentmethod_",
    )
    if any(any(x.startswith(p) for p in ohe_like) for x in keys):
        raise ValueError(
            "O corpo parece estar no formato da matriz **pós-transformação** (ex.: colunas tipo "
            "`internetservice_Fiber optic`, `contract_One year`). O endpoint /predict espera o nível **pré-ColumnTransformer**: "
            "o mesmo de `train_features_pre_transform.csv`, **sem** colunas derivadas da strategy "
            "(ex.: sem `is_new_customer`, `tenure_log`) e com valores **cruos** em `monthlycharges`/`totalcharges`, "
            "não normalizados. Ver documentação em `ChurnFeaturesInput`."
        )


def _prepare_prediction_features(run: PipelineRuns, domain: str, features: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame de entrada como no treino:
    - feature_engineering: chaves minúsculas (como `FeatureEngineering.load_data`) + strategy.build
    - baseline: chaves como enviadas (mesmos nomes que no CSV de treino) + clean/encode
    """
    ptype = run.pipeline_type
    d = domain.strip().lower()

    if ptype == "feature_engineering":
        from services.pipelines.feature_strategies import STRATEGY_REGISTRY

        if d not in STRATEGY_REGISTRY:
            raise ValueError(f"Domínio {domain!r} sem strategy. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")
        row = {str(k).strip().lower(): v for k, v in features.items()}
        row.pop("target", None)
        _infer_train_matrix_payload(row)
        df = pd.DataFrame([row])
        strategy = STRATEGY_REGISTRY[d]()
        _hydrate_strategy_from_run_metrics(strategy, run)
        strategy.validate(df)
        return strategy.build(df)

    if ptype == "baseline":
        row = {str(k).strip(): v for k, v in features.items()}
        row.pop("target", None)
        df = pd.DataFrame([row])
        return _baseline_clean_encode_predict(df)

    raise ValueError(f"pipeline_type não suportado para predição: {ptype!r}")


def _ts_from_baseline_model_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    m = re.search(
        r"baseline_model_[^_]+_(\d{8}_\d{6})(?:_[A-Za-z][A-Za-z0-9_]*)?\.joblib$",
        os.path.basename(model_path),
    )
    return m.group(1) if m else None


def _global_preprocessed_manifest() -> str:
    return os.path.abspath(os.path.join(settings.path_data_preprocessed, "manifest.json"))


async def _resolve_fe_manifest(session: AsyncSession, objective: str) -> str:
    """
    1) Tenta ``pre_processed/manifest.json``.
    2) Se falhar: baseline ``completed``, ``active`` e mesmo ``objective`` na BD —
       manifest ao lado do ``baseline_sample`` do run ou em ``PATH_DATA/PATH_LOGS/<ts>/``.
    """
    obj = objective.strip().lower()

    cand_global = _global_preprocessed_manifest()
    if os.path.isfile(cand_global):
        logger.info("FE: usando manifest global em %s", cand_global)
        return cand_global

    stmt = (
        select(PipelineRuns)
        .where(
            PipelineRuns.pipeline_type == "baseline",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            func.lower(PipelineRuns.objective) == obj,
        )
        .order_by(PipelineRuns.completed_at.desc(), PipelineRuns.id.desc())
        .limit(1)
    )
    res = await session.execute(stmt)
    run = res.scalars().one_or_none()
    if not run:
        raise ValueError(
            f"Manifest em falta: não há ``pre_processed/manifest.json`` e nenhum baseline activo para "
            f"objective={objective!r}. Rode baseline (API até ganhar recall ou modo sem defer)."
        )

    resolved: str | None = None

    outp = run.csv_output_path or ""
    if outp and os.path.isfile(outp):
        snap_dir = os.path.dirname(os.path.abspath(outp))
        try:
            sm, _ = _resolve_snapshot_manifest_and_sample(snap_dir)
            resolved = os.path.abspath(sm)
        except FileNotFoundError:
            pass

    if not resolved:
        ts = _ts_from_baseline_model_path(run.model_path)
        if ts:
            try:
                sm, _ = _resolve_snapshot_manifest_and_sample(
                    os.path.join(settings.path_data, settings.path_logs, ts)
                )
                resolved = os.path.abspath(sm)
            except FileNotFoundError:
                pass

    if not resolved:
        raise FileNotFoundError(
            f"Baseline activo (pipeline_run_id={run.id}) sem manifest encontrado no disco. "
            f"Espere Paths ``csv_output_path`` / modelo com timestamp compatíveis com PATH_DATA/PATH_LOGS.",
        )

    logger.info(
        "FE: manifest global ausente — fallback pelo baseline activo (run_id=%s) em %s",
        run.id,
        resolved,
    )
    return resolved


async def _resolve_fe_manifest_isolated_session(objective: str) -> str:
    """Consulta rápida com sessão própria (antes de iniciar run FE na sessão principal)."""
    from core.database import Session as SessionFactory

    session = SessionFactory()
    try:
        path = await _resolve_fe_manifest(session, objective)
        await session.commit()
        return path
    finally:
        await session.close()


def _publish_global_baseline_from_snapshot(run_timestamp: str) -> None:
    """
    Copia ``baseline_sample*.csv`` + ``manifest*.json`` do snapshot para ``pre_processed/``,
    ajustando no JSON o campo ``output_sample_csv_stable``. Suporta sufixos (ex.: ``_automatic``).
    """
    snap_root = os.path.join(settings.path_data, settings.path_logs, run_timestamp.strip())
    src_manifest, src_sample = _resolve_snapshot_manifest_and_sample(snap_root)

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    dst_sample = os.path.abspath(os.path.join(settings.path_data_preprocessed, "baseline_sample.csv"))
    shutil.copy2(src_sample, dst_sample)

    with open(src_manifest, encoding="utf-8") as f:
        man = json.load(f)
    man["output_sample_csv_stable"] = dst_sample
    man["manifest_snapshot"] = os.path.abspath(src_manifest)
    dst_manifest = os.path.abspath(os.path.join(settings.path_data_preprocessed, "manifest.json"))
    with open(dst_manifest, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    logger.info("Contrato FE global atualizado a partir do snapshot %s.", snap_root)


async def run_baseline(file: UploadFile, objective: str, user_id: int, db: AsyncSession) -> PipelineRuns:
    """Salva o CSV enviado, executa o Baseline e persiste o resultado."""
    from services.pipelines.baseline import Baseline
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    os.makedirs(settings.path_data, exist_ok=True)
    input_path = os.path.join(settings.path_data, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="baseline",
        objective=objective,
        status="processing",
        original_filename=file.filename,
        is_airflow_run=False,
    )

    async with db as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

        try:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(settings.path_data, settings.path_logs, run_ts)
            setup_pipeline_run_logging(
                snapshot_path,
                run_ts,
                run_id=run.id,
                objective=objective,
                pipeline_type="baseline",
            )
            from services.pipelines.feature_strategies import get_class_labels
            from services.pipelines.binary_decision_threshold import labels_from_probability_threshold
            pipeline = Baseline(
                pobjective=objective,
                run_timestamp=run_ts,
                csv_path=input_path,
                class_labels=get_class_labels(objective),
                defer_global_preprocess_contract=True,
            )
            pipeline.run(start_time=datetime.now())
            pipeline.save_artifacts()

            model_path = pipeline.baseline_model_joblib_path()
            csv_path = os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)

            model = pipeline.model
            y_pred_test = labels_from_probability_threshold(model, pipeline.x_test, pipeline.decision_threshold)
            y_proba_test = model.predict_proba(pipeline.x_test)[:, 1]
            _zd = {"zero_division": 0}
            yt = pipeline.y_test
            metrics = {
                "classification_decision_threshold": float(pipeline.decision_threshold),
                "test_accuracy": float(accuracy_score(yt, y_pred_test)),
                "test_f1": float(f1_score(yt, y_pred_test, **_zd)),
                "test_precision": float(precision_score(yt, y_pred_test, **_zd)),
                "test_recall": float(recall_score(yt, y_pred_test, **_zd)),
            }
            if int(yt.sum()) > 0 and int(len(yt) - yt.sum()) > 0:
                metrics["test_pr_auc"] = float(average_precision_score(yt, y_proba_test))

            run.status = "completed"
            run.model_path = model_path
            run.csv_output_path = csv_path
            run.completed_at = utcnow()
            run.metrics = metrics

            await _baseline_recall_winner(session, run, run_ts)

            metrics["baseline_fe_contract_published"] = bool(run.active)
            run.metrics = metrics

        except Exception as e:
            logger.error(f"Baseline falhou: {e}")
            run.status = "failed"
            run.error_message = str(e)[:1000]
            run.completed_at = utcnow()
            run.active = False

        session.add(run)
        await session.commit()
        await session.refresh(run)

    return run


async def run_feature_engineering(
    objective: str,
    user_id: int,
    db: AsyncSession,
    optimization_metric: str = "accuracy",
    min_precision: float | None = None,
    min_roc_auc: float | None = None,
    tuning_n_iter: int | None = None,
    time_limit_minutes: int = 2,
    acc_target: float | None = None,
    decision_threshold: float | None = None,
) -> tuple[PipelineRuns, str | None]:
    """
    Executa apenas o pipeline de Feature Engineering usando o contrato
    produzido pelo baseline (manifest + sample), sem rerun do baseline.
    """
    import tempfile

    from services.pipelines.feature_engineering import FeatureEngineering
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY
    from services.pipelines.fe_model_selection import normalize_optimization_metric
    from services.processor.artifact_bundle import safe_rmtree, safe_unlink
    from services.processor.deployment_service import get_active_deployment

    metric = normalize_optimization_metric(optimization_metric)
    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    resolved_manifest_path = await _resolve_fe_manifest_isolated_session(objective)
    with open(resolved_manifest_path, "r", encoding="utf-8") as f:
        baseline_manifest = json.load(f)

    manifest_objective = str(baseline_manifest.get("objective", "")).strip().lower()
    if manifest_objective != objective:
        raise ValueError(
            f"Manifest do baseline com objective='{manifest_objective}', mas FE foi solicitado para '{objective}'."
        )
    if baseline_manifest.get("sample_schema") != "raw_clean":
        raise ValueError(
            f"sample_schema inválido no manifest: {baseline_manifest.get('sample_schema')!r}. Esperado: 'raw_clean'."
        )
    csv_baseline = baseline_manifest.get("output_sample_csv_stable")
    if not csv_baseline:
        raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")
    csv_baseline = os.path.abspath(csv_baseline)
    if not os.path.isfile(csv_baseline):
        raise FileNotFoundError(f"CSV do baseline não encontrado: {csv_baseline}")

    baseline_input_path = baseline_manifest.get("input_csv_snapshot") or baseline_manifest.get("input_csv_source")
    baseline_input_path = os.path.abspath(baseline_input_path) if baseline_input_path else None
    # Contrato FE: o treino segue sempre ``output_sample_csv_stable`` (ex. pre_processed/baseline_sample.csv).
    # Gravamos esse ficheiro como etiqueta na BD; o CSV “upstream” do baseline fica só nas métricas de auditoria.
    fe_training_csv_basename = os.path.basename(csv_baseline)
    original_filename = fe_training_csv_basename

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="feature_engineering",
        objective=objective,
        status="processing",
        original_filename=original_filename,
        is_airflow_run=False,
    )
    zip_path: str | None = None
    run_root: str | None = None

    async with db as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

    effective_tuning_minutes = min(time_limit_minutes, settings.sync_fe_tune_max_minutes)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(settings.path_data, settings.path_logs, f"{run_ts}_fe{run.id}")

    try:
        setup_pipeline_run_logging(
            snapshot_path, run_ts, run_id=run.id, objective=objective, pipeline_type="feature_engineering"
        )

        run_root = tempfile.mkdtemp(prefix=f"fe_bundle_{run.id}_")
        prepare_fe_bundle_baseline_tree(
            run_root,
            resolved_manifest_path=resolved_manifest_path,
            baseline_manifest=baseline_manifest,
        )

        strategy = STRATEGY_REGISTRY[objective]()
        fe_d = os.path.join(run_root, "20_feature_engineering")
        fe_plots = os.path.join(fe_d, "plots")
        pipeline = FeatureEngineering(
            objective=objective,
            strategy=strategy,
            run_timestamp=run_ts,
            csv_path=csv_baseline,
            manifest_path=resolved_manifest_path,
            optimization_metric=metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            export_figures_dir=fe_plots,
            decision_threshold=decision_threshold,
            is_airflow_run=False,
        )
        pipeline.run(time_limit_minutes=effective_tuning_minutes, acc_target=acc_target)

        finalize_fe_bundle_pipeline_outputs(run_root, pipeline)
        fe_joblib = pipeline.fe_sklearn_joblib_path()

        active_dep = None
        baseline_ref = None
        async with db as s2:
            ad = await get_active_deployment(objective, s2)
            if ad is not None:
                active_dep = ad.id
            baseline_ref = await fetch_active_baseline_metrics_snapshot(s2, objective)

        merged_metrics: dict = {
            "fe_training_csv": csv_baseline,
            "fe_training_csv_basename": fe_training_csv_basename,
            "baseline_upstream_input_basename": (
                os.path.basename(baseline_input_path) if baseline_input_path else None
            ),
            "baseline_upstream_input_path": baseline_input_path,
            "baseline_manifest_ref": {
                "run_timestamp": baseline_manifest.get("run_timestamp"),
                "sample_schema": baseline_manifest.get("sample_schema"),
            },
            "optimization_metric": metric,
            "min_precision": min_precision,
            "min_roc_auc": min_roc_auc,
            "tuning_n_iter": tuning_n_iter if tuning_n_iter is not None else 100,
            "manifest_path_used": resolved_manifest_path,
            "tuning_time_limit_effective_minutes": effective_tuning_minutes,
            "tuning_time_limit_requested": time_limit_minutes,
        }
        if baseline_ref:
            merged_metrics["baseline_reference_metrics"] = baseline_ref
        _bcs = float(pipeline.best_cv_score)
        merged_metrics["best_cv_score"] = _bcs
        if math.isfinite(_bcs):
            merged_metrics[f"cv_{metric}"] = _bcs
        med = getattr(pipeline.strategy, "monthly_median", None)
        if med is not None:
            try:
                merged_metrics[_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY] = float(med)
            except (TypeError, ValueError):
                pass
        merged_metrics.update(dict(pipeline.tuned_metrics))
        merged_metrics.update(dict(pipeline.guardrails_summary))
        attach_mlp_metrics_snapshot(merged_metrics, pipeline)
        attach_fe_model_comparison_table(merged_metrics, pipeline)
        if pipeline.best_model_name:
            merged_metrics["best_model_name"] = pipeline.best_model_name

        zip_path = os.path.join(tempfile.gettempdir(), f"fe_artifacts_{run.id}_{run_ts}.zip")
        merged_metrics["fe_snapshot_dirname"] = f"{run_ts}_fe{run.id}"
        merged_metrics["fe_bundle_zip_filename"] = os.path.basename(zip_path)
        if os.path.isfile(zip_path):
            safe_unlink(zip_path)
        write_fe_manifest_zip_from_run_root(
            run_root,
            pipeline_run_id=run.id,
            objective=objective,
            run_timestamp=run_ts,
            csv_baseline=csv_baseline,
            original_filename=original_filename,
            merged_metrics=merged_metrics,
            mlflow_fe_run_id=pipeline.mlflow_run_id,
            best_model_name=pipeline.best_model_name,
            active_deployment_id=active_dep,
            zip_path=zip_path,
        )

        backend = "mlp" if settings.use_mlp_for_prediction else "sklearn"
        merged_metrics["inference_backend"] = backend
        merged_metrics["predict_model"] = "pytorch_mlp" if backend == "mlp" else "sklearn_pipeline"
        merged_metrics["sklearn_benchmark_classifier"] = pipeline.best_model_name
        if backend == "mlp" and pipeline.mlp_artifact_prefix:
            merged_metrics["mlp_artifact_prefix"] = pipeline.mlp_artifact_prefix

        run.status = "completed"
        run.model_path = fe_joblib
        run.csv_output_path = None
        run.metrics = merged_metrics
        run.completed_at = utcnow()
        run.inference_backend = backend

    except Exception as e:
        logger.error(f"Feature Engineering falhou: {e}", exc_info=True)
        run.status = "failed"
        run.error_message = str(e)[:1000]
        run.completed_at = utcnow()
        run.active = False
    finally:
        if run_root:
            safe_rmtree(run_root)

    async with db as session:
        merged = await session.merge(run)
        if merged.status == "completed" and merged.pipeline_type == "feature_engineering":
            await _fe_recall_winner(session, merged)
            m_final = dict(merged.metrics or {})
            m_final["fe_recall_champion"] = bool(merged.active)
            merged.metrics = m_final
            session.add(merged)
        await session.commit()
        await session.refresh(merged)

    return merged, zip_path


async def predict_for_domain(
    domain: str, features: dict, user_id: int, db: AsyncSession
) -> tuple[Predictions, InferenceReport, dict | None]:
    """Resolve deployment activo, infere e devolve predição + relatório.

    Retorno extra (3º elemento): ``recommended_items`` quando ``problem_type=recommendation``.
    """
    import domains  # noqa: F401
    import executors_ring  # noqa: F401 — engines tabular + recommendation

    from ml_core_ring.domain_plugin import get_domain
    from services.processor.deployment_service import NoActiveDeploymentError, get_active_deployment

    plugin = get_domain(domain)

    async with db as session:
        deployment = await get_active_deployment(domain, session)
        if not deployment:
            raise NoActiveDeploymentError(
                f"Nenhum modelo ativo para o domínio '{domain}'. Promova um pipeline concluído (admin)."
            )

        run = deployment.pipeline_run

        if plugin.problem_type == "recommendation":
            from ml_core_ring.artifact_manifest import ArtifactManifest
            from ml_core_ring.inference_engine import get_engine

            manifest = ArtifactManifest.from_pipeline_run(run)
            engine = get_engine(manifest)
            df_input = pd.DataFrame([features])
            result = engine.predict(df_input)
            item_ids = list(result.item_ids or [])

            pred = Predictions(
                user_id=user_id,
                pipeline_run_id=run.id,
                input_data=features,
                prediction=int(item_ids[0]) if item_ids else 0,
                probability=None,
            )
            session.add(pred)
            await session.commit()
            await session.refresh(pred)
            report = build_recommendation_inference_report(dict(run.metrics or {}))
            return pred, report, {"recommended_items": item_ids}

        backend = (run.inference_backend or "sklearn").strip().lower()
        df_input = _prepare_prediction_features(run, domain, features)

        from ml_core_ring import engines  # noqa: F401
        from ml_core_ring.artifact_manifest import ArtifactManifest
        from ml_core_ring.inference_engine import get_engine

        manifest = ArtifactManifest.from_pipeline_run(run)
        engine = get_engine(manifest)
        result = engine.predict(df_input)
        prediction_value = int(result.label if result.label is not None else 0)
        probability = result.probability

        pred = Predictions(
            user_id=user_id,
            pipeline_run_id=run.id,
            input_data=features,
            prediction=prediction_value,
            probability=probability,
        )

        session.add(pred)
        await session.commit()
        await session.refresh(pred)
        report = build_inference_report(dict(run.metrics or {}), backend)

    return pred, report, None
