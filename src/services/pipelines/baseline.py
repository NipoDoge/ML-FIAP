import logging
import mlflow
import pandas as pd
import os
import json
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from core.graphs import Graphs as gr
from core.configs import settings
from core.custom_logger import setup_log
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import re

import shutil

from ml_core_ring.mlflow_setup import configure_mlflow_tracking, ensure_mlflow_experiment

from services.pipelines.binary_decision_threshold import labels_from_probability_threshold
from services.utils import filename_with_suffix, log_training_csv_to_active_run

load_dotenv()
configure_mlflow_tracking()

#Enviroments
ppath_data = settings.path_data
ppath_data_preprocessed = settings.path_data_preprocessed
ppath_model = settings.path_model
ppath_graphs = settings.path_graphs

ptest_size = settings.test_size
prandom_state = settings.random_state

logger = logging.getLogger("ml.pipeline")
pmsg_raise = "Pipeline interrompido"
pagora = datetime.now().strftime('%Y%m%d_%H%M%S')
psnapshot_path = os.path.join(settings.path_data, settings.path_logs, pagora)

class Baseline:
    """
    Pipeline de baseline para problemas de **classificação binária tabulada**.

    Contrato de entrada do CSV
    --------------------------
    - A coluna alvo (target) deve ser sempre a **última coluna** do arquivo.
    - O arquivo **não deve conter coluna de ID** (remover antes do upload).
    - Valores ausentes nas **features** são tratados **após o split**, dentro de um
      ``sklearn.pipeline.Pipeline`` (mediana para numéricos, moda para categóricos),
      evitando vazamento de estatísticas do conjunto de teste.
    - O target é binarizado: qualquer valor > 0 vira 1, 0 permanece 0.

    Escopo e limitações
    --------------------
    - Suporta apenas **classificação binária**. Regressão e multiclasse
      estão fora do escopo deste pipeline.
    - Desbalanceamento de classes é tratado via `class_weight='balanced'`
      na Regressão Logística — estratégia de reamostragem não implementada.
    - Modelo gerado serve como referência mínima; comparar sempre com o
      pipeline de Feature Engineering antes de promover para produção.
    """
    def __init__(
        self,
        pobjective,
        run_timestamp: str | None = None,
        csv_path: str | None = None,
        class_labels: tuple[str, str] | None = None,
        *,
        defer_global_preprocess_contract: bool = False,
        artifact_name_suffix: str | None = None,
        decision_threshold: float | None = None,
    ):
        """
        Parameters
        ----------
        pobjective : str
            Identificador do domínio (ex.: "heart_disease", "churn").
        run_timestamp : str, optional
            Timestamp do run para naming de artefatos. Gerado automaticamente se omitido.
        csv_path : str, optional
            Caminho explícito do CSV (upload via API). Obrigatório para execução determinística.
        class_labels : tuple[str, str], optional
            Rótulos semânticos das classes (negativa, positiva).
            Padrão: ("Sem <objective>", "<objective>").
            Exemplo para churn: ("Não Churn", "Churn").
        defer_global_preprocess_contract : bool
            Quando ``True``, ``baseline_sample.csv`` + ``manifest.json`` só na pasta snapshot;
            o chamador deve copiar para ``pre_processed`` se o run vencer comparador recall.
        artifact_name_suffix : str, optional
            Inserido antes da extensão em contratos, joblib e nomes de gráficos (ex.: no Airflow
            usar ``_automatic`` para distinguir runs orquestrados de runs manuais pela API).
        decision_threshold : float, optional
            P(classe positiva) mínima para classificar como positivo (precision/recall nos relatórios).
            Omite-se ``settings.classification_decision_threshold`` (env ``CLASSIFICATION_DECISION_THRESHOLD``).
        """
        self.path_data = ppath_data
        self.path_data_preprocessed = ppath_data_preprocessed
        self.path_model = ppath_model
        self.path_graphs = ppath_graphs
        self.msg_raise = pmsg_raise
        self.objective = pobjective
        self.test_size = ptest_size
        self.random_state = prandom_state
        self.threshold_numeric_coercion = 0.8
        self._explicit_csv_path = os.path.abspath(csv_path) if csv_path else None
        self._artifact_suffix = (artifact_name_suffix or "").strip()
        self.contract_input_name = filename_with_suffix("input.csv", self._artifact_suffix)
        self.contract_sample_name = filename_with_suffix("baseline_sample.csv", self._artifact_suffix)
        self.contract_manifest_name = filename_with_suffix("manifest.json", self._artifact_suffix)
        
        if class_labels is not None:
            self.label_neg, self.label_pos = class_labels
        else:
            self.label_neg = f"Sem {self.objective}"
            self.label_pos = str(self.objective)
        if run_timestamp is not None:
            self.now = run_timestamp
            self.snapshot_path = os.path.join(settings.path_data, settings.path_logs, run_timestamp)
        else:
            self.now = pagora
            self.snapshot_path = psnapshot_path
        self.defer_global_preprocess_contract = defer_global_preprocess_contract
        self.decision_threshold = (
            float(decision_threshold)
            if decision_threshold is not None
            else float(settings.classification_decision_threshold)
        )
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError(
                f"classification_decision_threshold deve estar em [0, 1]. Recebido: {self.decision_threshold!r}"
            )
        self.current_csv_path = None
        self.data = None
        self.data_encoded = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.ratio = None
        self.model = None
        self.mlflow_run_id: str | None = None
        self.feature_groups: dict[str, list[str]] | None = None

        logger.info(f"Objective: {self.objective}")
        logger.info(f"Random state: {self.random_state}")
        logger.info(f"Test size: {self.test_size}")
        logger.info("classification_decision_threshold: %s", self.decision_threshold)

    @property
    def _graph_run_stamp(self) -> str:
        """Carimbo em nomes de ficheiros de gráfico/CSV auxiliar (inclui sufixo tipo ``_automatic``)."""
        return f"{self.now}{self._artifact_suffix}"

    def baseline_model_joblib_path(self) -> str:
        """Caminho absoluto do joblib do baseline para este run (alinhado a ``save()``)."""
        name = filename_with_suffix(
            f"baseline_model_{self.objective}_{self.now}.joblib", self._artifact_suffix
        )
        return os.path.join(self.path_model, name)
    
    def load_data(self):
        """
        Carregamento dos dados, passo 1.
        Requer `csv_path` explícito para manter execução determinística.
        """
        logger.info("Iniciando Dataset...")

        if not self._explicit_csv_path:
            logger.error("csv_path é obrigatório para execução determinística do baseline.")
            raise ValueError("Informe csv_path explícito ao iniciar o baseline.")
        if not os.path.isfile(self._explicit_csv_path):
            logger.error(f"CSV não encontrado: {self._explicit_csv_path}")
            raise ValueError(self.msg_raise)

        self.current_csv_path = self._explicit_csv_path
        logger.info(f"Arquivo CSV (rota explícita): {self.current_csv_path}")
        self.data = pd.read_csv(self.current_csv_path)
        self.target = self.data.columns.to_list().pop()
        
        
        logger.debug(f"Dataset carregado {self.data.shape}")
        logger.debug(f"5 primeiras linhas {self.data.head(5)}")
        
    def summary_overview(self):
        """
        Visão geral do dataset inicial, passo 2
        """
        
        logger.info("Informações gerais do dataset:\n")
        logger.debug(self.data.info())
        logger.info("Colunas presentes no dataset:\n")
        logger.debug(self.data.columns)
        logger.info("Estatísticas inciiais do dataset:\n")
        logger.debug(self.data.describe())
        
    # ------------------------------------------------------
    # EDA - Análise exploratória de dados
    # ------------------------------------------------------
    
    def analyze_convert_columns_to_numeric(
        self,
        df: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Diagnostica colunas com dtype 'object' que podem ser convertidas para numérico (float).

        Estratégia (passe único por coluna):
        1. Normaliza cada valor via `normalize_values_by_column`
           (remove moeda, resolve separador decimal BR/US).
        2. Mede a taxa de conversão sobre os valores PREENCHIDOS
           (ignora NaN original — desacopla missing de falha de parse).
        3. Marca `Recomenda_converter = True` quando taxa >= threshold.

        """
        threshold = self.threshold_numeric_coercion if threshold is None else threshold
        columns = ['Coluna', 'Taxa_numerica', 'N_unicos', 'Exemplos', 'Recomenda_converter']
        rows = []

        for col in df.select_dtypes(include=['object']).columns:
            serie = df[col]
            preenchidos = int(serie.notna().sum())

            normalized = serie.map(self.normalize_values_by_column)
            converted = pd.to_numeric(normalized, errors='coerce')
            convertidos = int(converted.notna().sum())

            taxa = round(convertidos / preenchidos, 2) if preenchidos else 0.0

            rows.append({
                'Coluna': col,
                'Taxa_numerica': taxa,
                'N_unicos': int(serie.nunique(dropna=True)),
                'Exemplos': list(serie.dropna().unique()[:3]),
                'Recomenda_converter': taxa >= threshold,
            })

            logger.debug(
                f"[object->numeric] col={col} taxa={taxa} "
                f"preenchidos={preenchidos} convertidos={convertidos} "
                f"recomenda={taxa >= threshold}"
            )

        report = pd.DataFrame(rows, columns=columns)
        if report.empty:
            return report
        return report.sort_values(by='Taxa_numerica', ascending=False, ignore_index=True)

    @staticmethod
    def normalize_values_by_column(x):
        """
        Normaliza um valor escalar para float.

        Regras:
        - NaN/None são preservados como NaN.
        - Remove símbolos de moeda e espaços, preserva dígitos, vírgula, ponto e sinal.
        - '1.234,56' -> 1234.56  (formato BR com milhar)
        - '123,45'   -> 123.45   (decimal BR)
        - '1.234'    -> 1234     (heurística milhar quando último bloco tem 3 dígitos;
                                  ambíguo para decimais '1.234' — documentado como limitação).
        - Strings que falham no parse retornam None (NaN-equivalente).
        """
        if pd.isna(x):
            return x

        x = str(x).strip()
        if x == "":
            return None

        x = re.sub(r'[^\d,.-]', '', x)

        if ',' in x and '.' in x:
            x = x.replace('.', '').replace(',', '.')
        elif ',' in x:
            x = x.replace(',', '.')
        elif '.' in x:
            partes = x.split('.')
            if len(partes[-1]) == 3:
                x = x.replace('.', '')

        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    def coerce_object_columns_to_numeric(
        self,
        df: pd.DataFrame,
        relatorio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aplica a coerção das colunas recomendadas pelo relatório
        (`Recomenda_converter == True`) para tipo numérico.
        """
        if relatorio.empty:
            return df

        colunas = relatorio.loc[relatorio['Recomenda_converter'], 'Coluna'].tolist()
        if not colunas:
            return df

        df = df.copy()
        for col in colunas:
            dtype_antes = df[col].dtype
            normalized = df[col].map(self.normalize_values_by_column)
            numeric = pd.to_numeric(normalized, errors='coerce')

            non_null = numeric.dropna()
            is_integer_like = bool(len(non_null) and (non_null % 1 == 0).all())

            df[col] = numeric.astype('Int64') if is_integer_like else numeric

            nan_originais = int(df[col].isna().sum())
            logger.info(
                f"[coerce] col={col} dtype {dtype_antes} -> {df[col].dtype} "
                f"n_nan={nan_originais} integer_like={is_integer_like}"
            )

        return df

    def missing_identifier(self):
        """
        Inicialmente como terceiro passo, identificar os missings values
        """
        
        logger.info("Inicializando a identificação dos missings values")
        
        if self.target not in self.data.columns:
            logger.error("A Coluna target não esta presente na fonte de dados")
            raise ValueError(self.msg_raise)
        
        if self.data[self.target].isnull().any():
            null_count = self.data[self.target].isnull().sum()
            logger.error(f"Valores nulos ou faltantes foram encontrados na coluna target {null_count}")
            raise ValueError(self.msg_raise)
        
        report_convertion = self.analyze_convert_columns_to_numeric(self.data)
        logger.info(f"Relatório de conversão para numérico:\n{report_convertion}")

        if not report_convertion.empty:
            gr.build_report(
                g_type=1,  # BARH
                x_data=report_convertion['Coluna'],
                y_data=report_convertion['Taxa_numerica'] * 100,
                title="Distribuição de conversão por coluna",
                xlabel="Porcentagem (%)",
                filename=filename_with_suffix(f"convert_object_to_numeric_{self.now}.png", self._artifact_suffix),
                color="skyblue",
            )

        convertiveis = (
            report_convertion[report_convertion['Recomenda_converter']]
            if not report_convertion.empty
            else report_convertion
        )

        if not convertiveis.empty:
            logger.warning(
                f"Colunas com dtype object convertíveis para numérico "
                f"(taxa >= {self.threshold_numeric_coercion}): "
                f"{convertiveis['Coluna'].tolist()}"
            )
            self.data = self.coerce_object_columns_to_numeric(
                self.data, report_convertion
            )
            logger.info(
                "Coerção aplicada. Refinamentos por coluna serão feitos na etapa de FE."
            )

        missing = pd.DataFrame({
            "Coluna" : self.data.columns,
            "Missing_count" : self.data.isnull().sum(),
            "Missing_percentage" : ((self.data.isnull().sum() / len(self.data))*100).round(2)
        })
        
        missing = missing[missing["Missing_count"]>0].sort_values(by='Missing_percentage',ascending=False)
        
        if len(missing)==0:
            logger.info("Dataset não apresentou missing values")
        else:
            logger.debug(f"Missing values identificados na fonte de dados: {missing}")
            gr.build_report(
            g_type=1, # BARH
            x_data=missing['Coluna'],
            y_data=missing['Missing_percentage'],
            title="Distribuição de Missing Values por Coluna",
            xlabel="Porcentagem (%)",
            filename=filename_with_suffix(f"missing_values_{self.now}.png", self._artifact_suffix),
            color="skyblue"
            )
            
    def pre_processor_churn(self):
        """
        Ajustes específicos do *churn*: mapeia Yes/No e variantes para 0/1,
        imputa ``TotalCharges`` se existir, e uniformiza ``object``/``category``
        para texto após o *replace*.
        """
        if self.objective != "churn":
            return

        df = self.data.copy()

        df = df.replace(
            {
                "Yes": 1,
                "No": 0,
                "No internet service": 0,
                "No phone service": 0,
            }
        )

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].fillna(0)

        for c in df.select_dtypes(include=["object", "category"]).columns:
            s = df[c]
            df[c] = s.map(lambda v: v if pd.isna(v) else str(v))

        self.data = df

            
    def target_analysis(self):
        """
        Neste passo 4, analisamos a fonte de dados identificando a variável target
        """
        
        logger.info("Iniciando a análise da target")

        self.data.rename(columns={self.target:'target'}, inplace=True)
        logger.info("Variável target definida")
        
        logger.info("Convertendo variável target para binário")
        
        le = LabelEncoder()
        self.data['target'] = le.fit_transform(self.data['target'].astype(str))
        logger.info(f"Target codificada: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        if self.data['target'].isnull().any():
            null_count = self.data['target'].isnull().sum()
            logger.error(f"Valores nulos ou faltantes foram encontrados na coluna target {null_count}")
            raise ValueError(self.msg_raise)
        
        logger.info("Iniciando analise de balancemanto da variável target")
        
        self.data['target'] = self.data['target'].astype(int)
        self.data['target'] = np.where(self.data['target']>0,1,0)
        
        
        target_counts = self.data['target'].value_counts()
        target_percentages = self.data['target'].value_counts(normalize=True) * 100
        
        logger.info(f"Contagem\n {target_counts}")
        logger.info("\nPercentual:")
        for idx, pct in target_percentages.items():
            label = self.label_neg if idx == 0 else self.label_pos
            logger.info(f"{label} ({idx}): {pct:.2f}")

        gr.build_report(
            g_type=3,  # PIE
            x_data=target_counts.values,
            labels=[f"{self.label_neg} (0)", f"{self.label_pos} (1)"],
            title="Proporção da Variável Target",
            filename=filename_with_suffix(f"target_distribution_pie_{self.now}.png", self._artifact_suffix),
            color="coral"
        )
        gr.build_report(
            g_type=2,
            x_data=[self.label_neg, self.label_pos],
            y_data=target_counts.values,
            title="Distribuição Absoluta da Target",
            ylabel="Quantidade",
            filename=filename_with_suffix(f"target_distribution_bar_{self.now}.png", self._artifact_suffix),
            color="skyblue"
        )
            
        self.ratio = target_counts.min()/target_counts.max()
        logger.info(f"\n Ratio de Balancemanto: {self.ratio:.2f}")
        
        if self.ratio < 0.5:
            logger.warning(
                "Dataset desbalanceado (ratio=%.2f). "
                "Mitigação aplicada: class_weight='balanced' na Regressão Logística. "
                "Estratégias de reamostragem (SMOTE, undersampling) estão fora do escopo deste pipeline.",
                self.ratio,
            )
        else:
            logger.info("✓ Dataset razoavelmente balanceado.")
            
    def view_data(self):
        """
        Amostra CSV + EDA churn via :class:`core.graphs.Graphs` (padrão do projeto).
        """
        logger.info("Iniciando visualização dos dados...")
        os.makedirs(self.path_graphs, exist_ok=True)
        sample_path = os.path.join(
            self.path_graphs, filename_with_suffix(f"data_view_{self.now}.csv", self._artifact_suffix)
        )
        try:
            self.data.head(5).to_csv(sample_path, index=False)
            logger.info("Amostra salva em: %s", sample_path)
        except Exception as e:
            logger.warning("Não foi possível salvar amostra CSV: %s", e)

        if str(self.objective).lower() != "churn":
            return

        gr.build_churn_view_data_eda(
            self.data,
            self._graph_run_stamp,
            self.label_neg,
            self.label_pos,
            graph_root=self.path_graphs,
            model_pipeline=None,
        )

    def outlier_analysis(self):
        """
        Passo 5, identificar outliers se houverem.
        """
        logger.info("Iniciando análise de outliers...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
               
        gr.build_outliers_report(
            data=self.data, 
            numeric_cols=numeric_cols, 
            filename=filename_with_suffix(f"outliers_boxplot_{self.now}.png", self._artifact_suffix)
        )
            
       
    # ------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------   

    @staticmethod
    def _is_binary_column(series: pd.Series) -> bool:
        """
        Retorna True quando a coluna possui apenas valores binários (0/1), ignorando NaN.
        """
        non_null = series.dropna()
        if non_null.empty:
            return False
        unique_values = set(non_null.unique().tolist())
        return unique_values.issubset({0, 1, 0.0, 1.0, np.int8(0), np.int8(1)})

    def _classify_feature_columns(self, x_train: pd.DataFrame) -> dict[str, list[str]]:
        """
        Classifica colunas de features em três grupos explícitos:
        binárias, numéricas contínuas e categóricas.
        """
        categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()

        binary_cols: list[str] = []
        continuous_cols: list[str] = []

        for col in numeric_cols:
            if self._is_binary_column(x_train[col]):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        # validações mínimas de consistência sem duplicar regras em outros pontos
        duplicated = (
            set(binary_cols).intersection(continuous_cols)
            | set(binary_cols).intersection(categorical_cols)
            | set(continuous_cols).intersection(categorical_cols)
        )
        if duplicated:
            raise ValueError(f"Classificação inconsistente de colunas: {sorted(duplicated)}")

        total_grouped = len(binary_cols) + len(continuous_cols) + len(categorical_cols)
        if total_grouped == 0:
            raise ValueError("Nenhuma feature válida encontrada para modelagem.")

        grouped = {
            "binary": binary_cols,
            "continuous": continuous_cols,
            "categorical": categorical_cols,
        }
        self.feature_groups = grouped
        return grouped

    def _build_preprocess_transformer(
        self, x_train: pd.DataFrame, groups: dict[str, list[str]] | None = None
    ) -> ColumnTransformer:
        """
        Pré-processamento aprendido apenas no treino.
        - binárias: passthrough (sem transformação)
        - contínuas: StandardScaler (+ imputação mediana só se houver nulos)
        - categóricas: OneHotEncoder (+ imputação moda só se houver nulos)
        """
        groups = groups or self._classify_feature_columns(x_train)
        binary_cols = groups["binary"]
        continuous_cols = groups["continuous"]
        categorical_cols = groups["categorical"]

        transformers: list[tuple] = []
        if continuous_cols:
            has_nulls_continuous = bool(x_train[continuous_cols].isna().any().any())
            continuous_steps = []
            if has_nulls_continuous:
                continuous_steps.append(("imputer", SimpleImputer(strategy="median")))
            continuous_steps.append(("scaler", StandardScaler()))
            transformers.append(
                (
                    "num_scaled",
                    SkPipeline(continuous_steps),
                    continuous_cols,
                )
            )
        if binary_cols:
            has_nulls_binary = bool(x_train[binary_cols].isna().any().any())
            if has_nulls_binary:
                raise ValueError(
                    "Colunas binárias com nulos detectadas. "
                    "Como binárias estão em passthrough por definição, "
                    "preencha nulos antes do treino."
                )
            transformers.append(
                (
                    "num_binary_passthrough",
                    "passthrough",
                    binary_cols,
                )
            )
        if categorical_cols:
            has_nulls_categorical = bool(x_train[categorical_cols].isna().any().any())
            categorical_steps = []
            if has_nulls_categorical:
                categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            categorical_steps.append(
                (
                    "ohe",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                )
            )
            transformers.append(
                (
                    "cat",
                    SkPipeline(categorical_steps),
                    categorical_cols,
                )
            )
        if not transformers:
            raise ValueError(
                "Nenhuma coluna numérica ou categórica (object/category) para modelagem."
            )
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def prepare_modeling_frame(self):
        """
        Monta o DataFrame usado no split: mesmas colunas que no treino da API,
        sem imputação nem encoding global (isso ocorre no Pipeline após o split).
        """
        logger.info("Preparando frame de modelagem (sem imputação/encoding pré-split)...")

        df_clean = self.data.copy()

        if "dataset" in df_clean.columns:
            df_clean.drop(columns=["dataset"], inplace=True)
            logger.info("Coluna 'dataset' removida (metadado de origem).")

        self.data_encoded = df_clean
        logger.info(
            f"Frame de modelagem: {self.data_encoded.shape} "
            "(valores ausentes nas features serão tratados no treino do Pipeline)"
        )
        logger.debug(f"Colunas: {self.data_encoded.columns.tolist()}")

    def prepare_and_train(self):
        """
        Treino: ``ColumnTransformer`` ajusta-se só a ``x_train``; a LR
        ajusta-se a **``x_train`` já transformado** (matriz densa/numérica),
        não a ``x_train`` em bruto. O ``SkPipeline`` junta os dois passos para
        ``predict``/serialização, sem chamar de novo ``Pipeline.fit`` nele.
        """
        logger.info("Iniciando treino do baseline (Pipeline sklearn)...")

        groups = self._classify_feature_columns(self.x_train)
        logger.info(
            "Classificação de colunas | binárias=%s contínuas=%s categóricas=%s",
            len(groups["binary"]),
            len(groups["continuous"]),
            len(groups["categorical"]),
        )
        logger.debug("Colunas binárias: %s", groups["binary"])
        logger.debug("Colunas contínuas: %s", groups["continuous"])
        logger.debug("Colunas categóricas: %s", groups["categorical"])

        preprocess = self._build_preprocess_transformer(self.x_train, groups=groups)
        classifier = LogisticRegression(
            random_state=self.random_state, class_weight="balanced", max_iter=1000
        )

        experiment_name = f"{self.objective}_baseline"
        ensure_mlflow_experiment(experiment_name)
        with mlflow.start_run(run_name=f"baseline_{self.objective}") as _mlr:
            self.mlflow_run_id = _mlr.info.run_id
            mlflow.log_param("classification_decision_threshold", float(self.decision_threshold))
            log_training_csv_to_active_run(
                self.current_csv_path,
                df=self.data,
                dataset_name=f"{self.objective}_baseline_training_csv",
                context="training",
            )

            x_tr = preprocess.fit_transform(self.x_train)
            classifier.fit(x_tr, self.y_train)
            self.model = SkPipeline(
                [("preprocess", preprocess), ("classifier", classifier)]
            )

            y_pred_train = labels_from_probability_threshold(self.model, self.x_train, self.decision_threshold)
            y_pred_test = labels_from_probability_threshold(self.model, self.x_test, self.decision_threshold)
            y_proba_train = self.model.predict_proba(self.x_train)[:, 1]
            y_proba_test = self.model.predict_proba(self.x_test)[:, 1]

            _zd = {"zero_division": 0}
            metrics = {
                "classification_decision_threshold": float(self.decision_threshold),
                "train_accuracy": accuracy_score(self.y_train, y_pred_train),
                "train_precision": precision_score(self.y_train, y_pred_train, **_zd),
                "train_recall": recall_score(self.y_train, y_pred_train, **_zd),
                "train_f1": f1_score(self.y_train, y_pred_train, **_zd),
                "test_accuracy": accuracy_score(self.y_test, y_pred_test),
                "test_f1": f1_score(self.y_test, y_pred_test, **_zd),
                "test_precision": precision_score(self.y_test, y_pred_test, **_zd),
                "test_recall": recall_score(self.y_test, y_pred_test, **_zd),
            }
            if int(self.y_train.sum()) > 0 and int(len(self.y_train) - self.y_train.sum()) > 0:
                metrics["train_pr_auc"] = float(
                    average_precision_score(self.y_train, y_proba_train)
                )
            else:
                metrics["train_pr_auc"] = float("nan")
                logger.warning(
                    "train_pr_auc omitido: treino sem ambas as classes (estratificação?)."
                )
            if int(self.y_test.sum()) > 0 and int(len(self.y_test) - self.y_test.sum()) > 0:
                metrics["test_pr_auc"] = float(
                    average_precision_score(self.y_test, y_proba_test)
                )
            else:
                metrics["test_pr_auc"] = float("nan")
                logger.warning(
                    "test_pr_auc omitido: conjunto de teste sem ambas as classes (use stratify ou mais dados)."
                )

            overfitting = metrics["train_accuracy"] - metrics["test_accuracy"]

            if str(self.objective).lower() == "churn":
                try:
                    gr.build_lr_coeff_importance_bars(
                        self.model, self._graph_run_stamp, graph_root=self.path_graphs
                    )
                except Exception as e:
                    logger.warning("Importância LR (gráfico) não gerada: %s", e)

            try:
                gr.build_precision_recall_curve(
                    self.y_test, y_proba_test, self._graph_run_stamp, graph_root=self.path_graphs, split_label="test"
                )
                gr.build_precision_recall_curve(
                    self.y_train, y_proba_train, self._graph_run_stamp, graph_root=self.path_graphs, split_label="train"
                )
            except Exception as e:
                logger.warning("Curvas Precision–Recall não geradas: %s", e)

            mlflow.log_params(self.model.get_params())
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}
            )
            mlflow.log_metric("overfitting_gap", overfitting)

            if os.path.isdir(self.path_graphs):
                # ``artifact_path`` é o prefixo *dentro do run* no MLflow (não caminho no disco).
                mlflow.log_artifacts(self.path_graphs, artifact_path="graphs")

            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Train Accuracy:  {metrics['train_accuracy']:.4f}")
            logger.info(f"Train Precision: {metrics['train_precision']:.4f}")
            logger.info(f"Train Recall:    {metrics['train_recall']:.4f}")
            logger.info(f"Train F1:        {metrics['train_f1']:.4f}")
            if not np.isnan(metrics.get("train_pr_auc", float("nan"))):
                logger.info(f"Train PR-AUC:    {metrics['train_pr_auc']:.4f}")
            logger.info(f"Test Accuracy:   {metrics['test_accuracy']:.4f}")
            logger.info(f"Test F1:         {metrics['test_f1']:.4f}")
            logger.info(f"Test Precision:  {metrics['test_precision']:.4f}")
            logger.info(f"Test Recall:     {metrics['test_recall']:.4f}")
            if not np.isnan(metrics["test_pr_auc"]):
                logger.info(f"Test PR-AUC:     {metrics['test_pr_auc']:.4f}")
            logger.info(f"Overfitting:    {overfitting:.4f}")
            logger.info("Treino e logs concluídos.")
    
    def split_data(self):
        """
        Divisão treino/teste no frame bruto (pós-EDA); imputação e encoding ocorrem
        apenas dentro do Pipeline em ``prepare_and_train``.
        """

        logger.info("Iniciando split data...")

        x = self.data_encoded.drop(columns="target")
        y = self.data_encoded["target"]
        
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=self.test_size,random_state=self.random_state,stratify=y)
            
    def save(self):
        """
        Salva o modelo treinado e o sample de integração com FE:
        - sample raw limpo (sem OHE/scaler), alinhado ao contrato Baseline -> FE
        - manifest com metadados da execução
        """
        logger.info("Iniciando o salvamento dos artefatos...")

        sample_raw = (
            pd.concat(
                [
                    self.x_train.assign(target=self.y_train),
                    self.x_test.assign(target=self.y_test),
                ],
                axis=0,
            )
            .sort_index()
        )
        defer = self.defer_global_preprocess_contract
        os.makedirs(self.snapshot_path, exist_ok=True)
        sample_stable_path = os.path.join(self.path_data_preprocessed, self.contract_sample_name)
        sample_snapshot_path = os.path.join(self.snapshot_path, self.contract_sample_name)
        if defer:
            sample_raw.to_csv(sample_snapshot_path, index=False)
            logger.info(
                "Sample por run (defer global FE) → %s | %s",
                sample_snapshot_path,
                sample_raw.shape,
            )
        else:
            for path in [self.path_data_preprocessed]:
                os.makedirs(path, exist_ok=True)
            sample_raw.to_csv(sample_stable_path, index=False)
            logger.info(
                "Sample FE raw limpo (sem OHE/scaler) → %s | %s",
                sample_stable_path,
                sample_raw.shape,
            )
            logger.info("Sample FE estável atualizado em: %s", sample_stable_path)

        os.makedirs(self.path_model, exist_ok=True)
        joblib_path = self.baseline_model_joblib_path()
        joblib.dump(self.model, joblib_path)
        logger.info(f"Modelo salvo em: {joblib_path}")

        groups = self.feature_groups or {"binary": [], "continuous": [], "categorical": []}
        input_snapshot_path = os.path.join(self.snapshot_path, self.contract_input_name)
        manifest_snapshot_path = os.path.join(self.snapshot_path, self.contract_manifest_name)
        stable_for_manifest = sample_snapshot_path if defer else sample_stable_path
        manifest = {
            "objective": self.objective,
            "run_timestamp": self.now,
            "input_csv_source": self.current_csv_path,
            "input_csv_snapshot": input_snapshot_path,
            "output_sample_csv_stable": os.path.abspath(stable_for_manifest),
            "output_sample_csv_snapshot": os.path.abspath(sample_snapshot_path),
            "manifest_snapshot": os.path.abspath(manifest_snapshot_path),
            "model_path": os.path.abspath(joblib_path),
            "input_row_count": int(self.data.shape[0]) if self.data is not None else None,
            "sample_row_count": int(sample_raw.shape[0]),
            "sample_column_count": int(sample_raw.shape[1]),
            "sample_schema": "raw_clean",
            "feature_groups": groups,
            "graphs_dir": os.path.join(self.snapshot_path, "graphs"),
        }
        snapshot_manifest_abs = manifest_snapshot_path
        with open(snapshot_manifest_abs, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
        logger.info("Manifest da corrida salvo em: %s", snapshot_manifest_abs)

        if not defer:
            manifest_stable = os.path.join(self.path_data_preprocessed, self.contract_manifest_name)
            with open(manifest_stable, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
            logger.info("Manifest global (pre_processed) atualizado em: %s", manifest_stable)

        try:
            if self.mlflow_run_id:
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    mlflow.log_artifact(joblib_path, artifact_path="model_files")
                    mlflow_sample = sample_snapshot_path if defer else sample_stable_path
                    mlflow.log_artifact(mlflow_sample, artifact_path="data_contract")
                    mlflow.log_dict(manifest, self.contract_manifest_name)
                logger.info("Modelo, sample e manifest registrados no MLflow.")
            else:
                logger.warning("Run MLflow ausente; artefatos não foram registrados no tracking.")
        except Exception as e:
            logger.error(f"Erro ao registrar no MLflow: {e}")

    # ------------------------------------------------------
    # Orquestração por responsabilidade
    # ------------------------------------------------------

    def _run_ingestion_and_quality(self, start_time):
        """
        Responsabilidade: ingestão e qualidade inicial dos dados.
        """
        self.load_data()
        logger.debug(f"Dados carregados: {datetime.now() - start_time}")
        self.summary_overview()
        logger.debug(f"Overview gerado: {datetime.now() - start_time}")
        self.missing_identifier()
        logger.debug(f"Identificados Missings: {datetime.now() - start_time}")
        self.pre_processor_churn()
        logger.debug(f"Pre-processamento: {datetime.now() - start_time}")
        self.target_analysis()
        logger.debug(f"Análise da target: {datetime.now() - start_time}")

    def _run_eda_and_modeling_prep(self, start_time):
        """
        Responsabilidade: EDA leve e preparação do frame para modelagem.
        """
        self.outlier_analysis()
        logger.debug(f"Análise de outliers: {datetime.now() - start_time}")
        self.view_data()
        logger.debug(f"Visualização de dados: {datetime.now() - start_time}")
        self.prepare_modeling_frame()
        logger.debug(f"Frame de modelagem: {datetime.now() - start_time}")
        self.split_data()
        logger.debug(f"Split de dados: {datetime.now() - start_time}")

    def _run_training_and_persistence(self, start_time):
        """
        Responsabilidade: treino, avaliação e persistência de artefatos.
        """
        self.prepare_and_train()
        logger.debug(f"Treino do baseline: {datetime.now() - start_time}")
        self.save()
        logger.debug(f"Artefatos salvos: {datetime.now() - start_time}")

    def run(self, start_time):
        """
        Orquestrador principal: executa as etapas por responsabilidade.
        """
        self._run_ingestion_and_quality(start_time)
        self._run_eda_and_modeling_prep(start_time)
        self._run_training_and_persistence(start_time)
    
    def save_artifacts(self):
        """
        Arquiva o dataset original e os gráficos gerados no diretório de snapshot.
        Não remove o arquivo original da pasta de entrada.
        """
        logger.info(f"Arquivando artefatos em: {self.snapshot_path}")
        os.makedirs(self.snapshot_path, exist_ok=True)

        if self.current_csv_path and os.path.exists(self.current_csv_path):
            try:
                target_input = os.path.join(self.snapshot_path, self.contract_input_name)
                shutil.copy(self.current_csv_path, target_input)
                logger.info("CSV original copiado para snapshot: %s", target_input)
            except Exception as e:
                logger.error(f"Erro ao copiar o CSV original: {e}")

        sample_in_snap = os.path.join(self.snapshot_path, self.contract_sample_name)
        if os.path.isfile(sample_in_snap):
            logger.info("Sample já gravado nesta pasta snapshot (%s — skip cópia).", sample_in_snap)
        else:
            sample_src = os.path.join(self.path_data_preprocessed, self.contract_sample_name)
            if os.path.isfile(sample_src):
                try:
                    sample_dst = os.path.join(self.snapshot_path, self.contract_sample_name)
                    shutil.copy(sample_src, sample_dst)
                    logger.info("Sample copiado de pre_processed para snapshot: %s", sample_dst)
                except Exception as e:
                    logger.error("Erro ao copiar baseline_sample.csv: %s", e)

        manifest_in_snap = os.path.join(self.snapshot_path, self.contract_manifest_name)
        if os.path.isfile(manifest_in_snap):
            logger.info("Manifest já neste snapshot (%s — skip cópia).", manifest_in_snap)
        else:
            manifest_src = os.path.join(self.path_data_preprocessed, self.contract_manifest_name)
            if os.path.isfile(manifest_src):
                try:
                    manifest_dst = manifest_in_snap
                    shutil.copy(manifest_src, manifest_dst)
                    logger.info("Manifest copiado de pre_processed para snapshot: %s", manifest_dst)
                except Exception as e:
                    logger.error("Erro ao copiar manifest.json: %s", e)

        target_graphs_path = os.path.join(self.snapshot_path, "graphs")
        if os.path.exists(self.path_graphs):
            if not os.path.exists(target_graphs_path):
                os.makedirs(target_graphs_path)
            
            for file_name in os.listdir(self.path_graphs):
                full_file_name = os.path.join(self.path_graphs, file_name)
                if os.path.isfile(full_file_name):
                    shutil.move(full_file_name, target_graphs_path)
            
            logger.info(f"Gráficos movidos para: {target_graphs_path}")
        
            
if __name__=="__main__":
            
    logger = setup_log(psnapshot_path, pagora)
    start_time = datetime.now()
    logger.info(f"Iniciando o pipeline: {start_time}")
    csv_path = os.getenv("BASELINE_CSV_PATH")
    if not csv_path:
        raise ValueError("Defina BASELINE_CSV_PATH para executar o baseline no modo CLI.")

    pipeline = Baseline(pobjective="heart_disease", csv_path=csv_path)
    try:
        pipeline.run(start_time)
        pipeline.save_artifacts()
        logger.debug(f"Artefatos salvos: {datetime.now() - start_time }")
        end_time = datetime.now() - start_time
        logger.debug(f"Baseline encerrado em : {end_time}")
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        raise ValueError("Pipeline interrompido devido a um erro.")
    
    

        
        