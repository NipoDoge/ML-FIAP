"""
Relatório de inferência para respostas de ``/predict``.

Centraliza leitura de ``PipelineRuns.metrics`` (gravadas no FE) para o consumidor
saber **o que está a ser servido** (sklearn vs MLP) e **métricas de holdout do treino**
(≠ métricas desta linha de predição).
"""

from __future__ import annotations

import math
from typing import Any

from core.configs import settings
from platform_ring.schemas.contracts import (
    BaselinePredictBlock,
    ComparisonPredict,
    ExperimentationPredict,
    InferenceReport,
    MetricSnapshot,
    PyTorchMLPExperiment,
    ServedModelPredict,
    TrainingSelectionSummaryPredict,
)


def attach_mlp_metrics_snapshot(merged_metrics: dict, pipeline: Any) -> None:
    """Acrescenta métricas de teste/validação do MLP a ``merged_metrics`` se o treino as tiver."""
    res = getattr(pipeline, "mlp_torch_result", None)
    if res is None:
        return
    merged_metrics["mlp_metrics_test"] = {k: float(v) for k, v in res.metrics_test.items()}
    merged_metrics["mlp_metrics_val"] = {k: float(v) for k, v in res.metrics_val.items()}
    training: dict[str, Any] = {
        "best_epoch": int(res.best_epoch),
        "best_val_loss": float(res.best_val_loss),
    }
    hp = getattr(pipeline, "mlp_torch_hparams", None) or {}
    for key in (
        "hidden_dims",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "max_epochs",
        "early_stopping_patience",
        "val_fraction",
    ):
        if key not in hp:
            continue
        val = hp[key]
        if key == "hidden_dims" and isinstance(val, (tuple, list)):
            training[key] = [int(x) for x in val]
        else:
            training[key] = val
    merged_metrics["mlp_training"] = training


def attach_fe_model_comparison_table(merged_metrics: dict, pipeline: Any) -> None:
    """Serializa a tabela sklearn pré/pós-tuning + MLP (treinamento / hold-out) tal como no ``fe_export``."""
    builder = getattr(pipeline, "_build_model_comparison_table", None)
    if not callable(builder):
        return
    df = builder()
    if df is None or getattr(df, "empty", True):
        return
    rows = []
    for rec in df.round(6).to_dict(orient="records"):
        row = {}
        for k, v in rec.items():
            if v is None:
                row[k] = None
            elif (
                isinstance(v, (float, int))
                and isinstance(v, float)
                and (math.isnan(v) or math.isinf(v))
            ):
                row[k] = None
            elif isinstance(v, (float, int)):
                row[k] = float(v) if isinstance(v, float) else int(v)
            else:
                row[k] = str(v)
        rows.append(row)
    merged_metrics["fe_model_comparison_table"] = rows


def _snapshot_from_sklearn_tuned(m: dict) -> MetricSnapshot | None:
    if not any(k in m for k in ("Acurácia", "Precisão", "Recall", "F1", "ROC AUC")):
        return None

    def _f(key: str) -> float | None:
        v = m.get(key)
        if v is None:
            return None
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return MetricSnapshot(
        accuracy=_f("Acurácia"),
        precision=_f("Precisão"),
        recall=_f("Recall"),
        f1=_f("F1"),
        roc_auc=_f("ROC AUC"),
    )


def _snapshot_from_mlp_test(d: Any) -> MetricSnapshot | None:
    if not isinstance(d, dict) or not d:
        return None

    def _g(key: str) -> float | None:
        v = d.get(key)
        if v is None:
            return None
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return MetricSnapshot(
        accuracy=_g("accuracy"),
        precision=_g("precision"),
        recall=_g("recall"),
        f1=_g("f1"),
        roc_auc=_g("roc_auc"),
    )


def _snapshot_from_baseline_run_metrics(br: dict) -> MetricSnapshot | None:
    """Métricas de teste do pipeline Baseline (chaves ``test_*`` na BD)."""

    def _g(*keys: str) -> float | None:
        for key in keys:
            v = br.get(key)
            if v is None:
                continue
            try:
                x = float(v)
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                continue
        return None

    acc = _g("test_accuracy")
    prec = _g("test_precision")
    rec = _g("test_recall")
    f1v = _g("test_f1")
    roc = _g("test_roc_auc")
    if all(x is None for x in (acc, prec, rec, f1v, roc)):
        return None
    return MetricSnapshot(accuracy=acc, precision=prec, recall=rec, f1=f1v, roc_auc=roc)


def build_inference_report(metrics: dict | None, inference_backend: str) -> InferenceReport:
    m = dict(metrics or {})
    backend = (inference_backend or m.get("inference_backend") or "sklearn").strip().lower()
    if backend not in ("sklearn", "mlp"):
        backend = "sklearn"

    predict_model_key = str(
        m.get("predict_model") or ("pytorch_mlp" if backend == "mlp" else "sklearn_pipeline")
    )
    sk_benchmark = m.get("sklearn_benchmark_classifier") or m.get("best_model_name")

    best_cv = m.get("best_cv_score")
    try:
        best_cv_f = (
            float(best_cv) if best_cv is not None and math.isfinite(float(best_cv)) else None
        )
    except (TypeError, ValueError):
        best_cv_f = None

    thr = m.get("classification_decision_threshold")
    try:
        thr_f = float(thr) if thr is not None else None
    except (TypeError, ValueError):
        thr_f = None

    sk_holdout = _snapshot_from_sklearn_tuned(m)
    mlp_holdout = _snapshot_from_mlp_test(m.get("mlp_metrics_test"))
    baseline_ref_raw = m.get("baseline_reference_metrics")
    baseline_ref = dict(baseline_ref_raw) if isinstance(baseline_ref_raw, dict) else None
    baseline_snap = _snapshot_from_baseline_run_metrics(baseline_ref) if baseline_ref else None

    comp_raw = m.get("fe_model_comparison_table")
    fe_rows: list[dict[str, Any]] = []
    if isinstance(comp_raw, list):
        fe_rows = [dict(r) for r in comp_raw if isinstance(r, dict)]

    served = mlp_holdout if backend == "mlp" else sk_holdout
    mlp_training = m.get("mlp_training") if isinstance(m.get("mlp_training"), dict) else None

    def _origem(row: dict[str, Any]) -> str:
        return str(row.get("Origem") or "")

    def _is_promoted_row(row: dict[str, Any]) -> bool:
        return "promovido" in _origem(row).lower()

    def _is_pre_tuning_row(row: dict[str, Any]) -> bool:
        return "pré-tuning" in _origem(row).lower()

    def _is_mlp_row(row: dict[str, Any]) -> bool:
        name = str(row.get("Modelo") or "").lower()
        return "pytorch" in name or name.endswith(" mlp")

    promoted_row = next((r for r in fe_rows if _is_promoted_row(r)), None)
    pre_tuning_rows = [r for r in fe_rows if _is_pre_tuning_row(r)]
    mlp_row = next((r for r in fe_rows if _is_mlp_row(r)), None)

    best_name = (m.get("best_model_name") or "").strip()
    if promoted_row is None and backend == "sklearn" and sk_holdout is not None and best_name:
        promoted_row = {
            "Modelo": f"{best_name} (tuned)",
            "Origem": "sklearn (pós-tuning, promovido)",
            "Accuracy": sk_holdout.accuracy,
            "Precision": sk_holdout.precision,
            "Recall": sk_holdout.recall,
            "F1": sk_holdout.f1,
            "ROC AUC": sk_holdout.roc_auc,
        }

    if backend == "mlp":
        model_used = dict(mlp_row) if mlp_row else (dict(promoted_row) if promoted_row else {})
        served_name = str(model_used.get("Modelo") or "PyTorch MLP")
        served_origin = str(
            model_used.get("Origem")
            or (
                "rede neural (treinamento; backend previsto para /predict após promote é MLP)"
                if settings.use_mlp_for_prediction
                else "rede neural (treinamento; não utilizada em inferência — promovido é o sklearn)"
            )
        )
    else:
        model_used = dict(promoted_row) if promoted_row else {}
        served_name = str(
            model_used.get("Modelo") or (f"{best_name} (tuned)" if best_name else predict_model_key)
        )
        served_origin = str(model_used.get("Origem") or "sklearn (pós-tuning, promovido)")

    exp_mlp: PyTorchMLPExperiment | None = None
    if mlp_row or mlp_training:
        exp_mlp = PyTorchMLPExperiment(
            holdout_row=dict(mlp_row) if mlp_row else {},
            training_summary=dict(mlp_training) if mlp_training else None,
        )

    comparison = ComparisonPredict(
        model_used_for_this_predict=model_used,
        pre_tuning_sklearn=pre_tuning_rows,
        experimentation=ExperimentationPredict(pytorch_mlp=exp_mlp),
    )

    baseline_block: BaselinePredictBlock | None = None
    if baseline_ref:
        cdf_bt = baseline_ref.get("classification_decision_threshold")
        try:
            cdf_bf = float(cdf_bt) if cdf_bt is not None and math.isfinite(float(cdf_bt)) else None
        except (TypeError, ValueError):
            cdf_bf = None

        baseline_block = BaselinePredictBlock(
            baseline_pipeline_run_id=baseline_ref.get("baseline_pipeline_run_id"),
            model_selection="Regressão Logística",
            role=baseline_ref.get("role"),
            classification_decision_threshold=cdf_bf,
            description=(
                "Métricas de **teste** do pipeline Baseline (sklearn, regressão logística + "
                "pré-processamento do contrato). Referência **antes** do feature engineering; "
                "não é o modelo servido em /predict."
            ),
            holdout_metrics={
                "accuracy": baseline_ref.get("test_accuracy"),
                "precision": baseline_ref.get("test_precision"),
                "recall": baseline_ref.get("test_recall"),
                "f1": baseline_ref.get("test_f1"),
                "pr_auc": baseline_ref.get("test_pr_auc"),
                "roc_auc": baseline_ref.get("test_roc_auc"),
            },
        )

    notes = [
        "O **Baseline** é sklearn simples (contrato + regressão logística). O **FE** acrescenta "
        "features, comparativo sklearn e MLP no mesmo run; o servido em `/predict` segue "
        "`inference_backend`.",
    ]

    if backend == "mlp":
        notes.append(
            "Inferência servida pela **MLP PyTorch** (bundle `.pt` + preprocess joblib). "
            "A probabilidade é σ(logit) da rede, alinhada ao pré-processador do FE."
        )
        if sk_holdout is None:
            notes.append(
                "Não há métricas de holdout do sklearn neste run para comparar lado-a-lado com a "
                "MLP no relatório; o classificador em `sklearn_classifier_from_cv_study` é "
                "referência do estudo de seleção."
            )
        else:
            notes.append(
                "Métricas sklearn no mesmo run (holdout) continuam disponíveis nas linhas do "
                "`comparison.pre_tuning_sklearn` e na linha promovida quando existir na tabela "
                "persistida."
            )
    else:
        notes.append("Inferência sklearn: probabilidade via `predict_proba` quando disponível.")

    notes.append(
        "Compare `baseline.holdout_metrics` com `holdout_metrics_served_model` para ver evolução "
        "pós-FE (atenção: thresholds de decisão podem diferir entre baseline 0,5 e FE 0,3)."
    )

    om = m.get("optimization_metric") or "métrica configurada"
    summary_lines = [
        f"Backend: **{backend}** · modelo servido: **{served_name}**.",
        (
            f"CV ({om}): **{best_cv_f:.4f}** · threshold métricas FE: **{thr_f:g}**."
            if best_cv_f is not None and thr_f is not None
            else (
                f"CV ({om}): **{best_cv_f:.4f}**."
                if best_cv_f is not None
                else (f"Threshold métricas FE: **{thr_f:g}**." if thr_f is not None else "")
            )
        ),
    ]
    summary_lines = [s for s in summary_lines if s]

    if baseline_snap and served:
        for label, bv_attr, sv_attr in (
            ("Recall", "recall", "recall"),
            ("F1", "f1", "f1"),
            ("ROC-AUC", "roc_auc", "roc_auc"),
        ):
            bv = getattr(baseline_snap, bv_attr)
            sv = getattr(served, sv_attr)
            if bv is not None and sv is not None:
                tag = "MLP" if backend == "mlp" else "FE sklearn"
                summary_lines.append(
                    f"Baseline vs servido ({tag}) — **{label}** (holdout): **{bv:.4f}** → **{sv:.4f}**."
                )
                break

    served_model = ServedModelPredict(
        inference_backend=backend,
        predict_model_key=predict_model_key,
        name=served_name,
        origin=served_origin,
    )

    training_summary = TrainingSelectionSummaryPredict(
        optimization_metric=m.get("optimization_metric"),
        best_cv_score=best_cv_f,
        classification_decision_threshold_for_holdout_metrics=thr_f,
        sklearn_classifier_from_cv_study=sk_benchmark,
    )

    return InferenceReport(
        served_model=served_model,
        training_selection_summary=training_summary,
        holdout_metrics_served_model=served,
        comparison=comparison,
        baseline=baseline_block,
        notes=notes,
        summary_lines=summary_lines,
    )


def build_recommendation_inference_report(metrics: dict | None) -> InferenceReport:
    """Relatório simplificado para ``/predict`` de recomendação."""
    m = dict(metrics or {})
    champion = str(m.get("champion_name") or "recommendation")
    served_model = ServedModelPredict(
        inference_backend="mlp",
        predict_model_key="recommendation_torch",
        name=champion,
        origin="run_promovido",
        label="Modelo de recomendação servido neste pedido",
    )
    notes = [
        "Saída: lista de item_id recomendados para o user_id pedido.",
        "Métricas abaixo referem-se ao holdout do treino (ndcg@k, etc.), não a este pedido.",
    ]
    if m.get("ndcg_at_k") is not None:
        notes.append(f"ndcg@k (treino): {m.get('ndcg_at_k')}")
    return InferenceReport(
        served_model=served_model,
        training_selection_summary=TrainingSelectionSummaryPredict(),
        comparison=ComparisonPredict(intro="Recomendação user-item (sem comparativo tabular)."),
        notes=notes,
        summary_lines=[f"Campeão treino: {champion}"],
    )
