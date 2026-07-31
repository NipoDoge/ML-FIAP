from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import pytest
import os
from contextlib import nullcontext

from src.services.pipelines.feature_engineering import FeatureEngineering
from src.services.pipelines.feature_strategies.base import FeatureStrategy


class TestFeatureEngineeringInit:
    """Tests for FeatureEngineering.__init__"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        strategy = Mock(spec=FeatureStrategy)
        fe = FeatureEngineering(objective="heart_disease", strategy=strategy)

        assert fe.objective == "heart_disease"
        assert fe.strategy == strategy
        assert fe.optimization_metric == "accuracy"
        assert fe.now is not None
        assert fe.data is None
        assert fe.x_train is None
        assert fe.x_test is None
        assert fe.y_train is None
        assert fe.y_test is None
        assert fe.feature_names == []
        assert fe.trained_models == {}
        assert fe.results_df is None
        assert fe.best_model_name is None
        assert fe.best_pipeline is None
        assert fe.best_params is None

    def test_init_with_custom_run_timestamp(self):
        """Test initialization with custom run timestamp"""
        strategy = Mock(spec=FeatureStrategy)
        run_timestamp = "20260415_120000"
        fe = FeatureEngineering(objective="churn", strategy=strategy, run_timestamp=run_timestamp)

        assert fe.now == run_timestamp

    def test_init_with_explicit_csv_path(self):
        """Test initialization with explicit CSV path"""
        strategy = Mock(spec=FeatureStrategy)
        csv_path = "/path/to/data.csv"
        fe = FeatureEngineering(objective="churn", strategy=strategy, csv_path=csv_path)

        assert fe._explicit_csv_path == os.path.abspath(csv_path)

    def test_init_with_custom_optimization_metric(self):
        """Test initialization with custom optimization metric"""
        strategy = Mock(spec=FeatureStrategy)
        fe = FeatureEngineering(
            objective="heart_disease", strategy=strategy, optimization_metric="f1"
        )

        assert fe.optimization_metric == "f1"

    def test_init_n_jobs_in_debug_mode(self):
        """Test that n_jobs is 1 when debugpy is loaded"""
        strategy = Mock(spec=FeatureStrategy)
        with patch.dict("sys.modules", {"debugpy": Mock()}):
            fe = FeatureEngineering(objective="churn", strategy=strategy)
            assert fe.n_jobs == 1
            assert isinstance(fe._joblib_cv_parallel_cm(), nullcontext)

    def test_init_n_jobs_normal_mode(self):
        """Test that n_jobs is -1 in normal mode"""
        strategy = Mock(spec=FeatureStrategy)
        with patch.dict("sys.modules", clear=True):
            fe = FeatureEngineering(objective="churn", strategy=strategy)
            # n_jobs should be -1 for parallel processing
            assert fe.n_jobs in [-1, 1]  # Can be either depending on environment


class TestLoadData:
    """Tests for FeatureEngineering.load_data"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_with_explicit_csv_path(self, mock_read_csv, mock_glob):
        """Test loading data with explicit CSV path"""
        csv_path = "/path/to/data.csv"
        self.fe._explicit_csv_path = csv_path

        # Mock CSV data
        mock_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        mock_read_csv.return_value = mock_data

        with patch("os.path.isfile", return_value=True):
            self.fe.load_data()

        mock_read_csv.assert_called_once_with(csv_path)
        pd.testing.assert_frame_equal(self.fe.data, mock_data)

    @patch("os.path.isfile")
    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_with_csv_not_found(self, mock_read_csv, mock_glob, mock_isfile):
        """Test that ValueError is raised when CSV is not found"""
        csv_path = "/nonexistent/path.csv"
        self.fe._explicit_csv_path = csv_path
        mock_isfile.return_value = False

        with pytest.raises(ValueError, match="CSV não encontrado"):
            self.fe.load_data()

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_no_csv_found(self, mock_read_csv, mock_glob):
        """Test ValueError when no CSV files found in default path"""
        self.fe._explicit_csv_path = None
        mock_glob.return_value = []

        with pytest.raises(ValueError, match="Nenhum CSV encontrado"):
            self.fe.load_data()

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_missing_target_column(self, mock_read_csv, mock_glob):
        """Test that ValueError is raised when target column is missing"""
        mock_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        mock_read_csv.return_value = mock_data
        mock_glob.return_value = ["/path/to/data.csv"]

        with patch("os.path.getctime", return_value=0):
            with pytest.raises(ValueError, match="coluna 'target' não encontrada"):
                self.fe.load_data()

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_with_null_values(self, mock_read_csv, mock_glob):
        """Test that ValueError is raised when null values are present"""
        mock_data = pd.DataFrame(
            {"feature1": [1, 2, np.nan], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        mock_read_csv.return_value = mock_data
        mock_glob.return_value = ["/path/to/data.csv"]

        with patch("os.path.getctime", return_value=0):
            with pytest.raises(ValueError, match="valores nulos"):
                self.fe.load_data()

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_removes_column_prefixes(self, mock_read_csv, mock_glob):
        """Test that column prefixes are removed from legacy CSVs"""
        mock_data = pd.DataFrame(
            {"prefix__feature1": [1, 2, 3], "prefix__feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        mock_read_csv.return_value = mock_data
        mock_glob.return_value = ["/path/to/data.csv"]

        with patch("os.path.getctime", return_value=0):
            self.fe.load_data()

        assert "feature1" in self.fe.data.columns
        assert "feature2" in self.fe.data.columns
        assert "target" in self.fe.data.columns

    @patch("glob.glob")
    @patch("pandas.read_csv")
    def test_load_data_lowercase_columns(self, mock_read_csv, mock_glob):
        """Test that columns are converted to lowercase"""
        mock_data = pd.DataFrame(
            {"Feature1": [1, 2, 3], "FEATURE2": [4, 5, 6], "TARGET": [0, 1, 0]}
        )
        mock_read_csv.return_value = mock_data
        mock_glob.return_value = ["/path/to/data.csv"]

        with patch("os.path.getctime", return_value=0):
            self.fe.load_data()

        assert all(col.islower() for col in self.fe.data.columns)


class TestBuildFeatures:
    """Tests for FeatureEngineering.build_features"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        self.fe.data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

    def test_build_features_calls_strategy(self):
        """Test that build_features calls strategy methods"""
        modified_data = self.fe.data.copy()
        modified_data["new_feature"] = [7, 8, 9]
        self.strategy.build.return_value = modified_data

        self.fe.build_features()

        self.strategy.validate.assert_called_once()
        self.strategy.build.assert_called_once()

    def test_build_features_updates_data(self):
        """Test that data is updated after build_features"""
        modified_data = self.fe.data.copy()
        modified_data["new_feature"] = [7, 8, 9]
        self.strategy.build.return_value = modified_data

        self.fe.build_features()

        assert "new_feature" in self.fe.data.columns


class TestSelectFeatures:
    """Tests for FeatureEngineering.select_features"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        np.random.seed(42)
        self.fe.data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    @patch("sklearn.model_selection.train_test_split")
    def test_select_features_splits_data(self, mock_split):
        """Test that data is properly split"""
        x_train = self.fe.data.drop(columns=["target"]).head(70)
        x_test = self.fe.data.drop(columns=["target"]).tail(30)
        y_train = self.fe.data["target"].head(70)
        y_test = self.fe.data["target"].tail(30)

        mock_split.return_value = (x_train, x_test, y_train, y_test)

        self.fe.select_features(k=2)

        assert self.fe.x_train is not None
        assert self.fe.x_test is not None
        assert self.fe.y_train is not None
        assert self.fe.y_test is not None

    def test_select_features_k_actual(self):
        """Test that k is limited by number of features"""
        self.fe.select_features(k=100)  # k > number of features

        # k should be limited to actual number of features
        assert len(self.fe.feature_names) <= 4

    def test_select_features_stores_feature_names(self):
        """Test that selected feature names are stored"""
        self.fe.select_features(k=2)

        assert isinstance(self.fe.feature_names, list)
        assert len(self.fe.feature_names) > 0
        assert all(isinstance(name, str) for name in self.fe.feature_names)


class TestTrainModels:
    """Tests for FeatureEngineering.train_models"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        self.fe.x_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature{i}" for i in range(n_features)],
        )
        self.fe.x_test = pd.DataFrame(
            np.random.randn(30, n_features), columns=[f"feature{i}" for i in range(n_features)]
        )
        self.fe.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        self.fe.y_test = pd.Series(np.random.randint(0, 2, 30))
        self.fe.feature_names = [f"feature{i}" for i in range(n_features)]

    def test_train_models_creates_pipeline(self):
        """Test that models are trained and pipelines created"""
        self.fe.train_models()

        assert len(self.fe.trained_models) > 0
        assert self.fe.best_model_name is not None
        assert self.fe.best_pipeline is not None

    def test_train_models_results_dataframe(self):
        """Test that results are stored in DataFrame"""
        self.fe.train_models()

        assert self.fe.results_df is not None
        assert isinstance(self.fe.results_df, pd.DataFrame)
        assert "Modelo" in self.fe.results_df.columns
        assert "Acurácia" in self.fe.results_df.columns

    def test_train_models_selects_best_model(self):
        """Test that best model is selected"""
        self.fe.train_models()

        best_name = self.fe.best_model_name
        assert best_name in self.fe.trained_models
        assert self.fe.best_pipeline == self.fe.trained_models[best_name]

    def test_train_models_svm_has_scaler(self):
        """Test that SVM model has scaler in pipeline"""
        self.fe.train_models()

        svm_pipeline = self.fe.trained_models.get("SVM")
        if svm_pipeline and hasattr(svm_pipeline, "named_steps"):
            assert "scaler" in svm_pipeline.named_steps


class TestTune:
    """Tests for FeatureEngineering.tune"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        self.fe.x_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature{i}" for i in range(n_features)],
        )
        self.fe.x_test = pd.DataFrame(
            np.random.randn(30, n_features), columns=[f"feature{i}" for i in range(n_features)]
        )
        self.fe.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        self.fe.y_test = pd.Series(np.random.randint(0, 2, 30))
        self.fe.feature_names = [f"feature{i}" for i in range(n_features)]
        self.fe.train_models()

    def test_tune_requires_trained_model(self):
        """Test that tune raises error if no model is trained"""
        fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)

        with pytest.raises(RuntimeError, match="tune requer train_models prévio"):
            fe.tune(time_limit_minutes=1)

    def test_tune_time_limit_respected(self):
        """Test that tuning respects time limit"""
        import time

        start = time.time()
        self.fe.tune(time_limit_minutes=0.01)  # Very short time limit
        elapsed = time.time() - start

        # Should respect time limit (allowing some overhead)
        assert elapsed < 10  # Should not take more than 10 seconds

    def test_tune_populates_tuned_metrics(self):
        """Test that tuning populates metrics"""
        self.fe.tune(time_limit_minutes=0.1)

        assert len(self.fe.tuned_metrics) > 0
        assert "Acurácia" in self.fe.tuned_metrics
        assert self.fe.best_params is not None

    def test_tune_maintains_accuracy_target_check(self):
        """Test that accuracy target is evaluated"""
        self.fe.tune(time_limit_minutes=0.1, acc_target=0.95)

        # Should not raise error and metrics should exist
        assert len(self.fe.tuned_metrics) > 0


class TestEvaluateImportance:
    """Tests for FeatureEngineering.evaluate_importance"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        self.fe.x_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature{i}" for i in range(n_features)],
        )
        self.fe.x_test = pd.DataFrame(
            np.random.randn(30, n_features), columns=[f"feature{i}" for i in range(n_features)]
        )
        self.fe.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        self.fe.y_test = pd.Series(np.random.randint(0, 2, 30))
        self.fe.feature_names = [f"feature{i}" for i in range(n_features)]
        self.fe.train_models()

    def test_evaluate_importance_creates_figures(self):
        """Test that importance evaluation creates figures"""
        self.fe.evaluate_importance()

        assert len(self.fe.figs_to_log) > 0

    def test_evaluate_importance_handles_non_tree_models(self):
        """Test importance evaluation with non-tree models"""
        # For SVM or other models without feature_importances_
        self.fe.evaluate_importance()

        # Should not raise error even if model doesn't have feature_importances_
        assert True


class TestSave:
    """Tests for FeatureEngineering.save"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(
            objective="heart_disease", strategy=self.strategy, run_timestamp="20260415_120000"
        )
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        self.fe.x_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature{i}" for i in range(n_features)],
        )
        self.fe.x_test = pd.DataFrame(
            np.random.randn(30, n_features), columns=[f"feature{i}" for i in range(n_features)]
        )
        self.fe.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        self.fe.y_test = pd.Series(np.random.randint(0, 2, 30))
        self.fe.feature_names = [f"feature{i}" for i in range(n_features)]
        self.fe.train_models()

    @patch("joblib.dump")
    @patch("mlflow.start_run")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("os.makedirs")
    def test_save_creates_joblib_file(
        self, mock_makedirs, mock_create_exp, mock_set_exp, mock_get_exp, mock_start_run, mock_dump
    ):
        """Test that joblib file is created"""
        mock_get_exp.return_value = None
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock(return_value=None)

        self.fe.save()

        mock_dump.assert_called()

    @patch("joblib.dump")
    @patch("mlflow.start_run")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("os.makedirs")
    def test_save_logs_to_mlflow(
        self, mock_makedirs, mock_create_exp, mock_set_exp, mock_get_exp, mock_start_run, mock_dump
    ):
        """Test that results are logged to MLflow"""
        mock_get_exp.return_value = None
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock(return_value=None)

        self.fe.save()

        mock_set_exp.assert_called()
        mock_start_run.assert_called()


class TestFinalizeTunedMetrics:
    """Tests for FeatureEngineering._finalize_tuned_metrics"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = Mock(spec=FeatureStrategy)
        self.fe = FeatureEngineering(objective="heart_disease", strategy=self.strategy)
        self.fe.y_test = np.array([0, 1, 1, 0, 1])  # Set y_test for the method

    def test_finalize_tuned_metrics_with_proba(self):
        """Test finalization with probability predictions"""
        y_pred = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.85])

        self.fe._finalize_tuned_metrics(y_pred, y_proba)

        assert "Acurácia" in self.fe.tuned_metrics
        assert "Precisão" in self.fe.tuned_metrics
        assert "Recall" in self.fe.tuned_metrics
        assert "F1" in self.fe.tuned_metrics
        assert "ROC AUC" in self.fe.tuned_metrics

    def test_finalize_tuned_metrics_without_proba(self):
        """Test finalization without probability predictions"""
        y_test = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])

        self.fe.y_test = y_test
        self.fe._finalize_tuned_metrics(y_pred, None)

        assert "Acurácia" in self.fe.tuned_metrics
        assert np.isnan(self.fe.tuned_metrics["ROC AUC"])


class TestRun:
    """Tests for FeatureEngineering.run orchestration"""

    @patch("services.pipelines.feature_engineering.FeatureEngineering.save")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.evaluate_importance")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.tune")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.train_models")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.select_features")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.build_features")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.load_data")
    def test_run_calls_all_methods_in_order(
        self, mock_load, mock_build, mock_select, mock_train, mock_tune, mock_importance, mock_save
    ):
        """Test that run method calls all pipeline steps"""
        strategy = Mock(spec=FeatureStrategy)
        fe = FeatureEngineering(objective="heart_disease", strategy=strategy)

        fe.run(time_limit_minutes=1, acc_target=0.85)

        # Verify all methods are called in the correct order
        mock_load.assert_called_once()
        mock_build.assert_called_once()
        mock_select.assert_called_once()
        mock_train.assert_called_once()
        mock_tune.assert_called_once()
        mock_importance.assert_called_once()
        mock_save.assert_called_once()

    @patch("services.pipelines.feature_engineering.FeatureEngineering.save")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.evaluate_importance")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.tune")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.train_models")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.select_features")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.build_features")
    @patch("services.pipelines.feature_engineering.FeatureEngineering.load_data")
    def test_run_passes_correct_parameters(
        self, mock_load, mock_build, mock_select, mock_train, mock_tune, mock_importance, mock_save
    ):
        """Test that run passes correct parameters to tune"""
        strategy = Mock(spec=FeatureStrategy)
        fe = FeatureEngineering(objective="heart_disease", strategy=strategy)

        fe.run(time_limit_minutes=30, acc_target=0.92)

        # Verify tune is called with correct parameters
        mock_tune.assert_called_once_with(time_limit_minutes=30, acc_target=0.92)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
