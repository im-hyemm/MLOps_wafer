"""예측 산출물, 실험 비교와 최종 checkpoint 평가."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CLASSES, ExperimentConfig, read_json, write_json
from .data import WaferDataset
from .models import build_model


def calculate_metrics(labels, predictions):
    """고정 9개 클래스 기준 원시 지표를 반환한다.

    Args:
        labels: 실제 클래스 ID 배열.
        predictions: 예측 클래스 ID 배열.

    Returns:
        반올림하지 않은 전체·클래스별 지표.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(9)),
        zero_division=0,
    )
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "worst_class_f1": float(f1.min()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1, weights=support)),
        # 단일 정답 다중 클래스에서 micro 지표는 accuracy와 같다.
        "micro_precision": accuracy,
        "micro_recall": accuracy,
        "micro_f1": accuracy,
        "support": int(support.sum()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }


@torch.no_grad()
def predict(model, loader, device, heartbeat=None):
    """순차 loader의 정답과 클래스 확률을 수집한다.

    Args:
        model: 학습된 모델.
        loader: shuffle=False인 DataLoader.
        device: PyTorch 장치.
        heartbeat: 배치 진행 상태 갱신 콜백.

    Returns:
        실제 라벨과 (샘플, 9) 예측 확률 배열의 튜플.
    """
    model.eval()
    labels, probabilities = [], []
    for batch, (images, features, targets) in enumerate(
        tqdm(loader, desc="예측 수집"), 1
    ):
        with torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images.to(device), features.to(device))
        labels.extend(targets.tolist())
        probabilities.append(logits.softmax(dim=1).float().cpu().numpy())
        if heartbeat:
            heartbeat(
                phase="saving_validation",
                batch=batch,
                total_batches=len(loader),
            )
    return np.asarray(labels), np.concatenate(probabilities)


def save_predictions(frame, labels, probabilities, output_dir, prefix=""):
    """예측 CSV와 confusion matrix를 저장하고 지표를 반환한다.

    Args:
        frame: 예측 순서와 같은 평가 DataFrame.
        labels: 실제 클래스 ID 배열.
        probabilities: (샘플, 9) 예측 확률 배열.
        output_dir: 저장 디렉터리.
        prefix: test_ 등 산출물 파일명 접두어.

    Returns:
        전체·클래스별 평가 지표.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    result = frame[["source_index", "lotName"]].copy()
    result["true_label"] = [CLASSES[index] for index in labels]
    result["predicted_label"] = [CLASSES[index] for index in predictions]
    for index, name in enumerate(CLASSES):
        result[f"prob_{name}"] = probabilities[:, index]
    result.to_csv(output_dir / f"{prefix}predictions.csv", index=False)
    matrix = confusion_matrix(labels, predictions, labels=list(range(9)))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure = Figure(figsize=(9, 8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        xticks=range(9),
        yticks=range(9),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title=f"{prefix or 'validation_'}confusion matrix",
    )
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    for i in range(9):
        for j in range(9):
            axis.text(
                j,
                i,
                f"{matrix[i, j]}\n{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalized[i, j] > 0.5 else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=140)
    figure.clear()
    return calculate_metrics(labels, predictions)


def evaluate_checkpoint(
    experiment_dir, splits, split="validation", device=None
):
    """같은 고정 분할의 checkpoint를 명시된 평가 세트에서 평가한다.

    Args:
        experiment_dir: 완료된 실험 디렉터리.
        splits: 원래 학습에 사용한 고정 분할.
        split: validation 또는 최종 확정 후 사용할 test.
        device: 선택적 PyTorch 장치.

    Returns:
        저장된 평가 지표.
    """
    if split not in ("validation", "test"):
        raise ValueError("validation 또는 test만 평가할 수 있습니다.")
    directory = Path(experiment_dir)
    if read_json(directory / "status.json")["state"] != "completed":
        raise ValueError("완료된 실험만 평가할 수 있습니다.")
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        directory / "model.pth", map_location=device, weights_only=True
    )
    if checkpoint["split_id"] != splits.split_id:
        raise ValueError("checkpoint와 평가 분할이 다릅니다.")
    config = ExperimentConfig(**checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    frame = getattr(splits, split)
    dataset = WaferDataset(
        frame, config, feature_stats=checkpoint["feature_stats"]
    )
    labels, probabilities = predict(
        model, DataLoader(dataset, batch_size=config.batch_size), device
    )
    metrics = save_predictions(
        frame, labels, probabilities, directory, prefix=f"{split}_"
    )
    write_json(directory / f"{split}_metrics.json", metrics)
    return metrics


def load_experiment_results(artifact_root):
    """완료된 실험의 validation 비교표를 반환한다.

    Args:
        artifact_root: 실험 디렉터리가 저장된 루트.

    Returns:
        실험 설정과 validation 성능을 포함한 DataFrame.
    """
    rows = []
    for path in sorted(Path(artifact_root).glob("*/metrics.json")):
        status = path.parent / "status.json"
        if status.exists() and read_json(status).get("state") == "completed":
            metrics = read_json(path)
            config = read_json(path.parent / "config.json")
            rows.append(
                {
                    "experiment_id": path.parent.name,
                    **config["config"],
                    "split_id": config["split_id"],
                    "code_version": config["code_version"],
                    "validation_macro_f1": metrics["validation"]["macro_f1"],
                    "worst_class_f1": metrics["validation"]["worst_class_f1"],
                    "parameters": metrics["parameters"],
                    "best_epoch": metrics["best_epoch"],
                }
            )
    return pd.DataFrame(rows)


def select_winner(results):
    """검증 F1과 복잡도를 기준으로 우승 행을 반환한다.

    Args:
        results: 동일 분할·코드의 완료 실험 비교 DataFrame.

    Returns:
        Macro-F1, 최저 클래스 F1, 적은 파라미터 순의 우승 행.
    """
    if results.empty:
        raise ValueError("비교할 완료 실험이 없습니다.")
    for field in ("split_id", "code_version"):
        if field in results and results[field].nunique() != 1:
            raise ValueError(
                f"서로 다른 {field}의 결과는 함께 선택할 수 없습니다."
            )
    return results.sort_values(
        [
            "validation_macro_f1",
            "worst_class_f1",
            "parameters",
            "experiment_id",
        ],
        ascending=[False, False, True, True],
    ).iloc[0]


def get_experiment_status(artifact_root):
    """실험 상태와 마지막 heartbeat의 경과 초를 표로 반환한다.

    Args:
        artifact_root: 실험 디렉터리가 저장된 루트.

    Returns:
        상태·단계·epoch·배치·마지막 갱신 시간을 담은 DataFrame.
    """
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for path in sorted(Path(artifact_root).glob("*/status.json")):
        row = read_json(path)
        row["seconds_since_update"] = (
            now - pd.Timestamp(row["updated_at"])
        ).total_seconds()
        rows.append(row)
    return pd.DataFrame(rows)
