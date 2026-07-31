import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

from core.configs import settings

logger = logging.getLogger(__name__)
path_graphs = settings.path_graphs


class Graphs:
    @staticmethod
    def _ensure_dir(base: str) -> None:
        if not os.path.exists(base):
            os.makedirs(base)
            logger.info("Diretório de gráficos criado: %s", base)

    @staticmethod
    def _save_fig(fig, base_dir: str, filename: str) -> str:
        Graphs._ensure_dir(base_dir)
        save_path = os.path.join(base_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Gráfico salvo em: %s", save_path)
        return save_path

    @staticmethod
    def build_report(
        g_type,
        x_data,
        y_data=None,
        title="",
        xlabel="",
        ylabel="",
        filename="graph.png",
        color="coral",
        labels=None,
    ):
        """
        Gera e salva gráficos (BAR, BARH, PIE).
        g_type: 1 (BARH), 2 (BAR), 3 (PIE)
        """
        Graphs._ensure_dir(path_graphs)

        fig, ax = plt.subplots(figsize=(10, 6))

        if g_type == 1:  # BARH
            ax.barh(x_data, y_data, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif g_type == 2:  # BAR
            ax.bar(x_data, y_data, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif g_type == 3:  # PIE
            ax.pie(x_data, labels=labels, autopct="%1.1f%%", colors=[color, "lightblue"])

        ax.set_title(title)
        plt.tight_layout()

        save_path = os.path.join(path_graphs, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Gráfico salvo em: %s", save_path)
        plt.close(fig)

    @staticmethod
    def build_outliers_report(data, numeric_cols, filename="outliers_analysis.png"):
        """
        Gera um grid de Boxplots com contagem de outliers via Z-Score.
        """
        num_cols_count = len(numeric_cols)
        nrows = (num_cols_count + 1) // 2
        ncols = 2 if num_cols_count > 1 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))

        if num_cols_count == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        idx = 0
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(x=data[col], ax=axes[idx], color="coral")

            col_data = data[col].dropna()
            z = np.abs(stats.zscore(col_data))
            outliers_count = (z > 3).sum()

            axes[idx].set_title(f"{col} | Outliers (Z>3): {outliers_count}")
            axes[idx].set_xlabel("")

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        Graphs._ensure_dir(path_graphs)
        save_path = os.path.join(path_graphs, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        logger.info("Relatório de outliers salvo: %s", save_path)
        plt.close(fig)

    # --- Churn EDA (Baseline / domínio churn) ---

    @staticmethod
    def _resolve_churn_target(s: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
        if "target" in s.columns:
            y = pd.to_numeric(s["target"], errors="coerce").fillna(0)
            return y.astype(int), "target"
        if "Churn" in s.columns:
            raw = s["Churn"].astype(str).str.strip().str.lower()
            y = raw.isin(("1", "true", "yes", "y")).astype(int)
            return y, "Churn"
        return None, None

    @staticmethod
    def _series_from_original_or_ohe(s: pd.DataFrame, base: str) -> pd.Series | None:
        if base in s.columns:
            return s[base].astype(str)
        prefix = f"{base}_"
        cols = [c for c in s.columns if c.startswith(prefix)]
        if not cols:
            return None
        try:
            sub = s[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            mx = sub.max(axis=1)
            idx = sub.idxmax(axis=1)
            lab = idx.astype(str).str[len(prefix) :]
            lab = lab.where(mx > 0, "(referência / drop_first)")
            return lab.astype(str)
        except Exception:
            return None

    @staticmethod
    def build_churn_view_data_eda(
        data: pd.DataFrame,
        run_stamp: str,
        label_neg: str,
        label_pos: str,
        graph_root: str | None = None,
        model_pipeline=None,
    ) -> None:
        """
        Conjunto de visualizações EDA para churn. Salva PNGs em ``graph_root`` (default: settings.path_graphs).
        ``model_pipeline``: se for um ``Pipeline`` sklearn já ajustado (preprocess+classifier), gera importância por |coef.|.
        """
        root = graph_root if graph_root is not None else path_graphs
        Graphs._ensure_dir(root)

        df = data.copy()
        y, _y_name = Graphs._resolve_churn_target(df)
        if y is None:
            logger.warning("Graphs.build_churn_view_data_eda: sem 'target' ou 'Churn'.")
            return
        df = df.assign(_y_churn=y)

        def _safe(name: str, fn) -> None:
            try:
                fn()
            except Exception as e:
                logger.warning("Gráfico %s não gerado: %s", name, e)

        def _save_name(fig, stem: str) -> None:
            Graphs._save_fig(fig, root, f"{stem}_{run_stamp}.png")

        def _cat_rate(cat_name: str) -> None:
            work = df.drop(columns=["_y_churn"], errors="ignore")
            ser = Graphs._series_from_original_or_ohe(work, cat_name)
            if ser is None:
                logger.warning("EDA churn: coluna ausente: %s", cat_name)
                return
            tmp = pd.DataFrame({cat_name: ser.values, "_y": df["_y_churn"].values})
            rate = tmp.groupby(cat_name, observed=False)["_y"].mean() * 100.0
            rate = rate.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4.0, len(rate) * 0.35)))
            rate.plot(kind="barh", ax=ax, color="steelblue")
            ax.axvline(
                df["_y_churn"].mean() * 100,
                color="coral",
                ls="--",
                lw=1,
                label="média",
            )
            ax.set_xlabel("Taxa de churn (%)")
            ax.set_title(f"Churn por {cat_name}")
            ax.legend(loc="lower right")
            _save_name(fig, f"churn_rate_{cat_name}")

        for col in ("Contract", "PaymentMethod", "InternetService"):
            _safe(f"churn_rate_{col}", lambda c=col: _cat_rate(c))

        def _tenure() -> None:
            if "tenure" not in df.columns:
                logger.warning("EDA churn: 'tenure' ausente.")
                return
            t = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
            bins = pd.cut(
                t,
                bins=[-0.001, 12, 24, 48, 72, 1e9],
                labels=["0–12", "13–24", "25–48", "49–72", "73+"],
            )
            tb = df.assign(_b=bins).groupby("_b", observed=False)["_y_churn"].mean() * 100.0
            fig, ax = plt.subplots(figsize=(8, 5))
            tb.plot(kind="bar", ax=ax, color="teal", rot=20)
            ax.set_ylabel("Taxa de churn (%)")
            ax.set_title("Churn por faixa de tenure")
            _save_name(fig, "churn_by_tenure_bin")

        _safe("churn_by_tenure_bin", _tenure)

        def _monthly() -> None:
            if "MonthlyCharges" not in df.columns:
                logger.warning("EDA churn: 'MonthlyCharges' ausente.")
                return
            mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            lab = df["_y_churn"].map({0: label_neg, 1: label_pos}).astype(str)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(x=lab, y=mc, ax=ax, inner="box", palette="muted")
            ax.set_title("MonthlyCharges por churn")
            ax.set_xlabel("")
            _save_name(fig, "monthly_charges_by_churn")

        _safe("monthly_charges_by_churn", _monthly)

        def _corr() -> None:
            num = df.select_dtypes(include=[np.number]).copy()
            if num.shape[1] < 2:
                logger.warning("EDA churn: poucas colunas numéricas para correlação.")
                return
            c = num.corr(numeric_only=True)
            fig, ax = plt.subplots(
                figsize=(
                    min(14, 0.35 * len(c.columns) + 4),
                    min(12, 0.35 * len(c.columns) + 4),
                )
            )
            sns.heatmap(
                c, ax=ax, cmap="RdBu_r", center=0, linewidths=0.2, annot=len(c) <= 18, fmt=".2f"
            )
            ax.set_title("Correlação (numéricas)")
            _save_name(fig, "corr_heatmap_numeric")

        _safe("corr_heatmap_numeric", _corr)

        def _inet_tech() -> None:
            inet = Graphs._series_from_original_or_ohe(df, "InternetService")
            tech = df["TechSupport"] if "TechSupport" in df.columns else None
            if tech is None:
                t2 = [c for c in df.columns if c.startswith("TechSupport_")]
                if t2:
                    tech = df[t2].apply(pd.to_numeric, errors="coerce").max(axis=1)
            if inet is None or tech is None:
                logger.warning("EDA churn: InternetService/TechSupport ausentes.")
                return
            tmp = pd.DataFrame(
                {
                    "InternetService": inet.astype(str),
                    "TechSupport": pd.to_numeric(tech, errors="coerce").fillna(0),
                }
            )
            tmp["_y"] = df["_y_churn"].values
            pv = tmp.pivot_table(
                index="InternetService", columns="TechSupport", values="_y", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pv * 100.0, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title("Churn (%) — InternetService × TechSupport")
            _save_name(fig, "churn_internet_x_techsupport")

        _safe("churn_internet_x_techsupport", _inet_tech)

        def _payment() -> None:
            ser = Graphs._series_from_original_or_ohe(df, "PaymentMethod")
            if ser is None:
                logger.warning("EDA churn: PaymentMethod ausente.")
                return
            tmp = pd.DataFrame({"PaymentMethod": ser.values, "_y": df["_y_churn"].values})
            rate = tmp.groupby("PaymentMethod", observed=False)["_y"].mean() * 100.0
            rate = rate.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(rate) * 0.35)))
            rate.plot(kind="barh", ax=ax, color="darkgreen")
            ax.set_xlabel("Taxa de churn (%)")
            ax.set_title("Churn por método de pagamento")
            _save_name(fig, "churn_by_payment_method")

        _safe("churn_by_payment_method", _payment)

        def _cxi() -> None:
            ctab = Graphs._series_from_original_or_ohe(df, "Contract")
            inet = Graphs._series_from_original_or_ohe(df, "InternetService")
            if ctab is None or inet is None:
                logger.warning("EDA churn: Contract/InternetService ausentes (heatmap).")
                return
            tmp = pd.DataFrame(
                {
                    "Contract": ctab.astype(str),
                    "InternetService": inet.astype(str),
                    "_y": df["_y_churn"].values,
                }
            )
            pv = tmp.pivot_table(
                index="Contract", columns="InternetService", values="_y", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(pv * 100.0, annot=True, fmt=".1f", cmap="Blues", ax=ax)
            ax.set_title("Churn (%) — Contract × InternetService")
            _save_name(fig, "churn_contract_x_internet")

        _safe("churn_contract_x_internet", _cxi)

        if model_pipeline is not None:
            _safe(
                "lr_coeff_importance",
                lambda: Graphs.build_lr_coeff_importance_bars(
                    model_pipeline, run_stamp, root, top_k=25
                ),
            )
        else:
            logger.debug(
                "EDA churn: sem modelo no view_data; importância LR será gerada após o treino."
            )

    @staticmethod
    def build_lr_coeff_importance_bars(
        pipeline,
        run_stamp: str,
        graph_root: str | None = None,
        top_k: int = 25,
        preprocess_step: str = "preprocess",
        classifier_step: str = "classifier",
    ) -> None:
        """|coeficientes| da LR após pré-processamento (top_k features)."""
        root = graph_root if graph_root is not None else path_graphs
        try:
            pre = pipeline.named_steps[preprocess_step]
            clf = pipeline.named_steps[classifier_step]
            names = pre.get_feature_names_out()
            coef = np.ravel(clf.coef_)
        except Exception as e:
            logger.warning("Importância LR: extração de coeficientes falhou: %s", e)
            return
        if len(names) != len(coef):
            logger.warning("Importância LR: tamanho de coef. incompatível com features.")
            return
        top = pd.Series(np.abs(coef), index=names).sort_values(ascending=False).head(top_k)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.25)))
        top.sort_values().plot(kind="barh", ax=ax, color="darkslateblue")
        ax.set_title(f"Importância relativa (|coef.| LR) — top {top_k}")
        ax.set_xlabel("|coeficiente|")
        Graphs._save_fig(fig, root, f"lr_coeff_importance_{run_stamp}.png")

    @staticmethod
    def build_precision_recall_curve(
        y_true,
        y_score,
        run_stamp: str,
        graph_root: str | None = None,
        split_label: str = "test",
    ) -> str | None:
        """
        Curva precision–recall (probabilidades da classe positiva) e AP (área sob a
        curva, equivalente ao que ``average_precision_score`` reporta para binário).
        """
        root = graph_root if graph_root is not None else path_graphs
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            logger.warning(
                "PR curve (%s): necessária pelo menos duas classes em y; omitido.",
                split_label,
            )
            return None
        ap = float(average_precision_score(y_true, y_score))
        fig, ax = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="LogisticRegression")
        ax.set_title(f"Precision–Recall ({split_label}) — AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return Graphs._save_fig(fig, root, f"precision_recall_{split_label}_{run_stamp}.png")
