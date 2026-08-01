import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

import services.pipelines.baseline as baseline_module
from services.pipelines.baseline import Baseline, gr


def test_init_default_labels():
    baseline = Baseline(pobjective="churn")

    assert baseline.objective == "churn"
    assert baseline.label_neg == "Sem churn"
    assert baseline.label_pos == "churn"
    assert baseline._explicit_csv_path is None
    assert baseline.current_csv_path is None
    assert baseline.data is None


def test_artifact_name_suffix_prepended_before_extension():
    b = Baseline(
        pobjective="churn",
        run_timestamp="20250101_120000",
        csv_path="/tmp/dummy.csv",
        artifact_name_suffix="_automatic",
    )
    assert b.contract_manifest_name == "manifest_automatic.json"
    assert b.contract_sample_name == "baseline_sample_automatic.csv"
    assert b.contract_input_name == "input_automatic.csv"
    assert b.baseline_model_joblib_path().endswith(
        "baseline_model_churn_20250101_120000_automatic.joblib"
    )


def test_init_custom_class_labels():
    baseline = Baseline(pobjective="heart_disease", class_labels=("No HD", "HD"))

    assert baseline.label_neg == "No HD"
    assert baseline.label_pos == "HD"


def test_load_data_with_explicit_path(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,target\n1,2,0\n3,4,1")

    baseline = Baseline(pobjective="target", csv_path=str(csv_file))
    baseline.load_data()

    assert baseline.data is not None
    assert baseline.current_csv_path == str(csv_file)
    assert baseline.target == "target"
    assert baseline.data.shape == (2, 3)


def test_load_data_explicit_path_not_found():
    baseline = Baseline(pobjective="target", csv_path="non_existent.csv")

    with pytest.raises(ValueError):
        baseline.load_data()


def test_load_data_no_files(tmp_path):
    baseline = Baseline(pobjective="target")
    baseline.path_data = str(tmp_path)

    with pytest.raises(ValueError):
        baseline.load_data()


def test_load_data_picks_latest(tmp_path):
    file1 = tmp_path / "old.csv"
    file2 = tmp_path / "new.csv"

    file1.write_text("a,b,target\n1,2,0")
    time.sleep(0.05)
    file2.write_text("a,b,target\n3,4,1")

    baseline = Baseline(pobjective="target", csv_path=str(file2))

    baseline.load_data()

    assert baseline.current_csv_path.endswith("new.csv")
    assert baseline.data.iloc[0].to_dict() == {"a": 3, "b": 4, "target": 1}


def test_summary_overview_does_not_raise():
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    baseline.target = "target"

    baseline.summary_overview()


def test_missing_identifier_passes_when_no_missing():
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    baseline.target = "target"

    baseline.missing_identifier()


def test_missing_identifier_raises_for_missing_target_column():
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame({"feature": [1, 2], "value": [0, 1]})
    baseline.target = "target"

    with pytest.raises(ValueError):
        baseline.missing_identifier()


def test_missing_identifier_raises_for_null_target_values():
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame({"feature": [1, 2], "target": [0, None]})
    baseline.target = "target"

    with pytest.raises(ValueError):
        baseline.missing_identifier()


def test_target_analysis_binary_encoding(monkeypatch):
    baseline = Baseline(pobjective="churn")
    baseline.data = pd.DataFrame({"feature": [1, 2, 3, 4], "status": ["no", "yes", "yes", "no"]})
    baseline.target = "status"

    calls = []
    monkeypatch.setattr(gr, "build_report", lambda **kwargs: calls.append(kwargs))

    baseline.target_analysis()

    assert "target" in baseline.data.columns
    assert set(baseline.data["target"].unique()) == {0, 1}
    assert baseline.ratio == 1.0
    assert len(calls) == 2


def test_outlier_analysis_calls_graph_builder(monkeypatch):
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    baseline.target = "target"

    captured = {}

    def fake_build_outliers_report(data, numeric_cols, filename):
        captured["data_shape"] = data.shape
        captured["numeric_cols"] = numeric_cols
        captured["filename"] = filename

    monkeypatch.setattr(gr, "build_outliers_report", fake_build_outliers_report)

    baseline.outlier_analysis()

    assert captured["data_shape"] == (3, 2)
    assert captured["numeric_cols"] == ["feature"]
    assert "outliers_boxplot_" in captured["filename"]


def test_clean_and_encode_imputes_and_encodes():
    baseline = Baseline(pobjective="target")
    baseline.data = pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "category": ["a", None, "b", "a"],
            "value": [1.0, None, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    )
    baseline.target = "target"

    baseline.prepare_modeling_frame()

    assert "dataset" not in baseline.data_encoded.columns
    assert "target" in baseline.data_encoded.columns
    assert baseline.data_encoded.shape[0] == 4
    assert baseline.data_encoded["value"].isna().sum() == 1


def test_split_data_creates_train_test_sets():
    baseline = Baseline(pobjective="target")
    baseline.data_encoded = pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "feature": [1, 2, 3, 4],
        }
    )
    baseline.test_size = 0.5
    baseline.random_state = 42

    baseline.split_data()

    assert baseline.x_train.shape[0] == 2
    assert baseline.x_test.shape[0] == 2
    assert set(baseline.y_train.unique()) <= {0, 1}
    assert set(baseline.y_test.unique()) <= {0, 1}


def test_prepare_and_train_trains_model(monkeypatch, tmp_path):
    baseline = Baseline(pobjective="target")
    baseline.x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    baseline.x_test = pd.DataFrame({"feature": [4.0, 5.0]})
    baseline.y_train = pd.Series([0, 0, 1, 1])
    baseline.y_test = pd.Series([0, 1])
    baseline.path_graphs = str(tmp_path / "graphs")

    metrics_record = {}

    class DummyRun:
        info = SimpleNamespace(run_id="test-run")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(
        baseline_module,
        "mlflow",
            SimpleNamespace(
                get_experiment_by_name=lambda name: None,
                create_experiment=lambda *args, **kwargs: "exp",
                set_experiment=lambda *args, **kwargs: None,
                start_run=lambda *args, **kwargs: DummyRun(),
                log_param=lambda name, value: metrics_record.setdefault(name, value),
                log_params=lambda params: metrics_record.setdefault("params", params),
                log_metrics=lambda metrics: metrics_record.setdefault("metrics", metrics),
                log_metric=lambda name, value: metrics_record.setdefault(name, value),
                log_artifacts=lambda *args, **kwargs: None,
                sklearn=SimpleNamespace(
                    log_model=lambda *args, **kwargs: metrics_record.setdefault("model_logged", True)
                ),
        ),
    )
    monkeypatch.setattr(baseline_module, "ensure_mlflow_experiment", lambda *args, **kwargs: None)

    baseline.prepare_and_train()

    assert baseline.model is not None
    assert hasattr(baseline.model, "predict")
    assert "metrics" in metrics_record
    assert "train_accuracy" in metrics_record["metrics"]
    assert "test_accuracy" in metrics_record["metrics"]


def test_save_writes_preprocessed_csv_and_model(monkeypatch, tmp_path):
    from sklearn.linear_model import LogisticRegression

    baseline = Baseline(pobjective="target")
    baseline.data_encoded = pd.DataFrame({"target": [0, 1], "feature": [1, 2]})
    baseline.model = LogisticRegression().fit(
        baseline.data_encoded[["feature"]], baseline.data_encoded["target"]
    )
    baseline.objective = "target"
    baseline.now = "now"
    baseline.path_data_preprocessed = str(tmp_path / "preprocessed")
    baseline.path_model = str(tmp_path / "models")
    baseline.snapshot_path = str(tmp_path / "snapshot")
    baseline.x_train = baseline.data_encoded[["feature"]]
    baseline.x_test = baseline.data_encoded[["feature"]]
    baseline.y_train = baseline.data_encoded["target"]
    baseline.y_test = baseline.data_encoded["target"]

    monkeypatch.setattr(
        baseline_module, "mlflow", SimpleNamespace(log_artifact=lambda *args, **kwargs: None)
    )

    baseline.save()

    assert os.path.exists(tmp_path / "preprocessed" / "baseline_sample.csv")
    assert any(
        path.name.startswith("baseline_model_target_now")
        for path in (tmp_path / "models").iterdir()
    )


def test_save_artifacts_moves_csv_and_graph_files(tmp_path):
    baseline = Baseline(pobjective="target")
    baseline.current_csv_path = str(tmp_path / "input.csv")
    with open(baseline.current_csv_path, "w") as handle:
        handle.write("feature,target\n1,0")

    baseline.path_graphs = str(tmp_path / "graphs")
    os.makedirs(baseline.path_graphs, exist_ok=True)
    graph_file = os.path.join(baseline.path_graphs, "graph.png")
    with open(graph_file, "w") as handle:
        handle.write("dummy")

    baseline.snapshot_path = str(tmp_path / "snapshot")
    os.makedirs(baseline.snapshot_path, exist_ok=True)

    baseline.save_artifacts()

    assert os.path.exists(baseline.current_csv_path)
    assert os.path.exists(os.path.join(baseline.snapshot_path, "input.csv"))
    assert os.path.exists(os.path.join(baseline.snapshot_path, "graphs", "graph.png"))
    assert not os.path.exists(graph_file)
