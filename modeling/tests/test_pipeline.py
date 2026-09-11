"""분할 누수, 입력 일관성 및 실제 CPU 학습 흐름 검증."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from modeling.config import (
    CLASSES,
    ExperimentConfig,
    atomic_replace,
    read_json,
)
from modeling.data import WaferDataset, create_or_load_splits, load_dataset
from modeling.evaluation import (
    calculate_metrics,
    get_experiment_status,
    load_experiment_results,
)
from modeling.features import FEATURE_NAMES, extract_geometric_features
from modeling.models import build_model
from modeling.preprocessing import encode_map, preprocess_map
from modeling.run_experiment import run_stage, stage_configs
from modeling.training import configure_logging, train_experiment


@pytest.fixture
def splits(tmp_path):
    """작은 9개 클래스 데이터를 실제 분할 함수로 준비한다."""
    torch.set_num_threads(1)
    configure_logging()
    rows = []
    for label, name in enumerate(CLASSES):
        for index in range(30):
            wafer = np.ones((18, 20), dtype=np.uint8)
            wafer[:, :2] = 0
            if name != "none":
                wafer[
                    2 + label : 4 + label, 3 + index % 10 : 5 + index % 10
                ] = 2
            rows.append(
                {
                    "waferMap": wafer,
                    "lotName": f"lot{index // 3}",
                    "failureType": np.array([[name]]),
                }
            )
    path = tmp_path / "data.pkl"
    pd.DataFrame(rows).to_pickle(path)
    frame = load_dataset(path)
    return create_or_load_splits(frame, tmp_path / "artifacts")


def test_splits_and_fingerprint(splits, tmp_path):
    """분할이 재현되고 행·라벨 분포가 유지되는지 확인한다."""
    frame = (
        pd.concat([splits.train, splits.validation, splits.test])
        .sort_values("source_index")
        .reset_index(drop=True)
    )
    repeated = create_or_load_splits(frame, tmp_path / "artifacts")
    assert splits.split_id == repeated.split_id
    assert (
        splits.train.source_index.tolist()
        == repeated.train.source_index.tolist()
    )
    groups = [
        set(getattr(splits, name).source_index)
        for name in ("train", "validation", "test")
    ]
    assert len(groups[0]) == 216 and len(groups[1]) == len(groups[2]) == 27
    assert (
        not groups[0] & groups[1]
        and not groups[0] & groups[2]
        and not groups[1] & groups[2]
    )
    assert splits.train.source_index.tolist() != sorted(
        splits.train.source_index
    )
    changed = frame.copy(deep=True)
    changed.at[changed.index[0], "waferMap"] = np.full(
        (18, 20), 2, dtype=np.uint8
    )
    assert (
        create_or_load_splits(changed, tmp_path / "changed").split_id
        != splits.split_id
    )


@pytest.mark.parametrize(
    "mode", ["fixed_resize", "resize_pad", "resize_pad_mask"]
)
def test_preprocessing(mode):
    """범주값과 실제 die mask가 보존되는지 확인한다."""
    source = np.array([[0, 1, 2, 0], [0, 1, 1, 0]], dtype=np.uint8)
    wafer = preprocess_map(source, mode, 16)
    encoded = encode_map(wafer, mode)
    assert wafer.shape == (16, 16) and set(np.unique(wafer)) <= {0, 1, 2}
    assert encoded.dtype == np.float32
    if mode == "resize_pad_mask":
        np.testing.assert_array_equal(encoded[1], wafer > 0)
        assert encoded.shape == (2, 16, 16)


def test_features_and_hybrid_augmentation(splits, monkeypatch):
    """증강 맵과 특징이 일치하고 평가가 train 통계를 재사용하는지 확인한다."""
    config = ExperimentConfig(
        model="hybrid_cnn", augmentation=True, target_size=16
    )
    dataset = WaferDataset(splits.train, config, training=True)
    monkeypatch.setattr(
        "modeling.data.augment_map",
        lambda image: np.ascontiguousarray(np.fliplr(image)),
    )
    image, feature, _ = dataset[0]
    reconstructed = (image[0].numpy() * 2).astype(np.uint8)
    expected = (
        extract_geometric_features(reconstructed)
        - dataset.feature_stats["mean"]
    ) / np.array(dataset.feature_stats["std"])
    np.testing.assert_allclose(feature.numpy(), expected, rtol=1e-5, atol=1e-5)
    validation = WaferDataset(
        splits.validation, config, feature_stats=dataset.feature_stats
    )
    assert validation.feature_stats == dataset.feature_stats
    constant = dataset.raw_features.std(axis=0) < 1e-6
    assert np.all(np.asarray(dataset.feature_stats["std"])[constant] == 1.0)
    assert len(FEATURE_NAMES) == 24
    for wafer in (np.zeros((8, 8)), np.ones((8, 8)), np.full((8, 8), 2)):
        assert np.isfinite(extract_geometric_features(wafer)).all()


@pytest.mark.parametrize(
    "model_name", ["small_cnn", "spatial_cnn", "residual_cnn", "hybrid_cnn"]
)
@pytest.mark.parametrize("channels", [1, 2])
def test_model_shapes(model_name, channels):
    """각 모델의 1/2채널 출력 계약을 확인한다."""
    config = ExperimentConfig(
        model=model_name,
        preprocessing="resize_pad_mask" if channels == 2 else "fixed_resize",
    )
    model = build_model(config)
    result = model(torch.zeros(2, channels, 16, 16), torch.zeros(2, 24))
    assert result.shape == (2, 9)


def test_eleven_runs_and_final_test_only(splits, tmp_path):
    """실제 11회 짧은 CPU 학습과 완료 재사용·최종 평가를 검증한다."""
    root = tmp_path / "experiments"
    base = ExperimentConfig(epochs=1, target_size=16, batch_size=64)
    with pytest.raises(ValueError, match="앞 단계"):
        stage_configs("recipes", splits, root, base)
    for stage, expected in (
        ("models", 4),
        ("recipes", 7),
        ("preprocessing", 9),
    ):
        run_stage(stage, splits, root, base, device="cpu")
        assert len(load_experiment_results(root)) == expected
        assert not list(root.glob("*/test_*.json"))
    prior_mtimes = {
        path: path.stat().st_mtime_ns for path in root.glob("*/model.pth")
    }
    paths = run_stage("final", splits, root, base, device="cpu")
    assert len(load_experiment_results(root)) == 11
    assert len(list(root.glob("*/test_metrics.json"))) == 3
    assert all(
        path.stat().st_mtime_ns == modified
        for path, modified in prior_mtimes.items()
    )
    final = read_json(next(root.glob("suites/*/final_summary.json")))
    assert [row["seed"] for row in final["runs"]] == [42, 43, 44]
    values = [row["macro_f1"] for row in final["runs"]]
    assert final["macro_f1_mean"] == pytest.approx(np.mean(values))
    assert final["macro_f1_std"] == pytest.approx(np.std(values, ddof=1))
    for path in paths:
        assert read_json(path / "status.json")["state"] == "completed"
        history = pd.read_csv(path / "history.csv")
        assert len(history) == 1
        details = read_json(path / "history_details.json")
        assert details["classes"] == list(CLASSES)
        for phase in ("train", "validation"):
            metrics = details["epochs"][0][phase]
            matrix = np.array(metrics["confusion_matrix"])
            assert matrix.shape == (9, 9)
            assert matrix.sum() == len(getattr(splits, phase))
            assert history.iloc[0][f"{phase}_accuracy"] == pytest.approx(
                matrix.trace() / matrix.sum()
            )
            for metric, value in metrics.items():
                if metric not in ("per_class", "confusion_matrix"):
                    assert history.iloc[0][
                        f"{phase}_{metric}"
                    ] == pytest.approx(value)
            for name, values in metrics["per_class"].items():
                for metric, value in values.items():
                    column = f"{phase}_{name}_{metric}"
                    assert history.iloc[0][column] == pytest.approx(value)
        assert "Epoch 1/1" in (path / "train.log").read_text(encoding="utf-8")
        checkpoint = torch.load(path / "model.pth", weights_only=True)
        model = build_model(ExperimentConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
    assert (get_experiment_status(root).seconds_since_update >= 0).all()


@pytest.mark.parametrize(
    "error,state",
    [
        (RuntimeError("테스트 오류"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_error_status_and_restart(splits, tmp_path, monkeypatch, error, state):
    """학습 오류·사용자 중단은 완료로 재사용하지 않고 재학습한다."""
    root = tmp_path / state
    config = ExperimentConfig(epochs=1, target_size=16, batch_size=128)
    import modeling.training as training

    original = training._run_epoch

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(training, "_run_epoch", fail)
    with pytest.raises(type(error)):
        train_experiment(config, splits, root, device="cpu")
    directory = next(root.glob("small_cnn_*"))
    assert read_json(directory / "status.json")["state"] == state
    assert "Traceback" in (directory / "train.log").read_text(encoding="utf-8")
    monkeypatch.setattr(training, "_run_epoch", original)
    train_experiment(config, splits, root, device="cpu")
    assert read_json(directory / "status.json")["state"] == "completed"


def test_early_stopping_and_logging(splits, tmp_path, monkeypatch):
    """개선 없는 epoch에서 조기 종료하고 로거를 중복 등록하지 않는다."""
    config = ExperimentConfig(
        epochs=5, early_stopping_patience=1, target_size=16, batch_size=128
    )
    metrics = calculate_metrics(list(range(9)), [0] * 9)
    metrics["confusion_matrix"] = [[1] + [0] * 8 for _ in range(9)]
    monkeypatch.setattr(
        "modeling.training._run_epoch", lambda *args, **kwargs: (1.0, metrics)
    )
    path = train_experiment(config, splits, tmp_path / "early", device="cpu")
    assert read_json(path / "metrics.json")["epochs_run"] == 2
    assert "Early stopping" in (path / "train.log").read_text(encoding="utf-8")
    logger = configure_logging(path / "train.log")
    logger = configure_logging(path / "train.log")
    assert len(logger.handlers) == 2


def test_metrics_imbalanced_and_absent_classes():
    """불균형·누락 클래스에서도 집계 지표가 정확한지 확인한다."""
    from sklearn.metrics import precision_recall_fscore_support

    labels, predicted = [0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]
    result = calculate_metrics(labels, predicted)
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted,
            labels=list(range(9)),
            average=average,
            zero_division=0,
        )
        for name, expected in zip(
            ("precision", "recall", "f1"), (precision, recall, f1)
        ):
            assert result[f"{average}_{name}"] == pytest.approx(expected)
    assert result["support"] == 6
    assert result["per_class"]["none"]["support"] == 0


def test_notebook_is_valid():
    """배포 notebook의 JSON과 셀 구성이 유효한지 확인한다."""
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    path = Path(__file__).parents[1] / "compare_experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(cell.source for cell in notebook.cells)
    stages = set()
    transformer = TransformerManager()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(transformer.transform_cell(cell.source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_stage"
            ):
                stages.add(node.args[0].value)
    assert stages == {"models", "recipes", "preprocessing", "final"}
    assert "def train" not in sources


def test_atomic_replace_retries_transient_lock(tmp_path, monkeypatch):
    """일시적 파일 잠금 후 교체를 재시도하는지 확인한다."""
    source = tmp_path / "status.json.tmp"
    destination = tmp_path / "status.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    original = Path.replace
    attempts = []

    def replace_with_lock(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("동기화 중 잠금")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace_with_lock)
    monkeypatch.setattr("modeling.config.time.sleep", lambda seconds: None)
    atomic_replace(source, destination)
    assert len(attempts) == 3
    assert destination.read_text(encoding="utf-8") == "new"
