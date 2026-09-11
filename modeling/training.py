"""진행률·파일 로그·상태 저장을 포함한 단일 실험 학습."""

import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    CLASSES,
    atomic_replace,
    experiment_identity,
    read_json,
    write_json,
)
from .data import WaferDataset
from .evaluation import calculate_metrics, predict, save_predictions
from .features import FEATURE_NAMES
from .models import build_model


def configure_logging(log_path=None):
    """이전 핸들러를 닫고 화면과 선택적 파일 로거를 구성한다.

    Args:
        log_path: 로그를 append할 경로. None이면 화면만 사용한다.

    Returns:
        중복 핸들러가 없는 modeling 로거.
    """
    logger = logging.getLogger("modeling")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


class ProgressStatus:
    """주기적으로 갱신하는 실험 상태 기록기."""

    def __init__(self, directory, experiment_id, stage):
        """실험 ID와 시작 시각을 보관한다.

        Args:
            directory: status.json 저장 디렉터리.
            experiment_id: 현재 실험 ID.
            stage: models 등 현재 단계 이름.
        """
        self.path = Path(directory) / "status.json"
        self.started = time.monotonic()
        self.last_write = -float("inf")
        self.data = {
            "experiment_id": experiment_id,
            "stage": stage,
            "state": "running",
            "epoch": 0,
            "batch": 0,
            "total_batches": 0,
            "phase": "starting",
        }

    def update(self, force=False, **values):
        """30초 간격 또는 강제 요청 시 상태를 원자적으로 저장한다.

        Args:
            force: 주기와 상관없이 즉시 저장할지 여부.
            **values: 현재 epoch·배치 등 갱신할 상태 필드.
        """
        self.data.update(values)
        now = time.monotonic()
        if force or now - self.last_write >= 30:
            self.data.update(
                updated_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=now - self.started,
            )
            write_json(self.path, self.data)
            self.last_write = now


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    status,
    epoch,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)
    phase = "train" if training else "validation"
    status.update(
        force=True,
        phase=phase,
        epoch=epoch,
        batch=0,
        total_batches=len(loader),
    )
    numerator, denominator = 0.0, 0.0
    labels, predictions = [], []
    progress = tqdm(
        loader,
        desc=f"{status.data['experiment_id']} epoch {epoch} {phase}",
        leave=False,
    )
    for batch, (images, features, targets) in enumerate(progress, 1):
        images, features, targets = (
            images.to(device),
            features.to(device),
            targets.to(device),
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(images, features)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{phase} loss가 유한하지 않습니다.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        weight = (
            targets.numel()
            if criterion.weight is None
            else criterion.weight[targets].sum().item()
        )
        numerator += loss.item() * weight
        denominator += weight
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        progress.set_postfix(
            loss=f"{numerator / denominator:.4f}", refresh=False
        )
        status.update(
            phase=phase,
            epoch=epoch,
            batch=batch,
            total_batches=len(loader),
            loss=numerator / denominator,
        )
    metrics = calculate_metrics(labels, predictions)
    metrics["confusion_matrix"] = confusion_matrix(
        labels, predictions, labels=list(range(len(CLASSES)))
    ).tolist()
    return numerator / denominator, metrics


def _flatten_epoch_metrics(phase, metrics):
    """전체·클래스별 epoch 지표를 CSV 컬럼으로 펼친다."""
    row = {
        f"{phase}_{name}": value
        for name, value in metrics.items()
        if name not in ("per_class", "confusion_matrix")
    }
    for class_name, values in metrics["per_class"].items():
        for name, value in values.items():
            row[f"{phase}_{class_name}_{name}"] = value
    return row


def train_experiment(
    config, splits, artifact_root, stage="single", force=False, device=None
):
    """단일 실험을 학습하고 validation 산출물을 저장한다.

    Args:
        config: 재현 가능한 실험 설정.
        splits: 모든 후보에서 공유하는 고정 분할.
        artifact_root: 결과 저장 루트.
        stage: 로그에 표시할 실험 단계.
        force: 완료 결과도 처음부터 재학습할지 여부.
        device: 선택적 CPU/CUDA 장치.

    Returns:
        완료된 실험 디렉터리.
    """
    experiment_id, metadata = experiment_identity(config, splits.split_id)
    directory = Path(artifact_root) / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(directory / "train.log")
    status_path = directory / "status.json"
    required = (
        "config.json",
        "metrics.json",
        "history.csv",
        "history_details.json",
        "predictions.csv",
        "confusion_matrix.png",
        "model.pth",
    )
    if (
        not force
        and status_path.exists()
        and read_json(status_path).get("state") == "completed"
    ):
        if (
            all((directory / name).exists() for name in required)
            and read_json(directory / "config.json") == metadata
        ):
            logger.info("완료 실험 재사용: %s", experiment_id)
            return directory
    status = ProgressStatus(directory, experiment_id, stage)
    status.update(force=True)
    try:
        write_json(directory / "config.json", metadata)
        # 재시작 시 이전 실행의 test 결과가 새 모델의 결과처럼 남지 않게 한다.
        for name in (
            "test_metrics.json",
            "test_predictions.csv",
            "test_confusion_matrix.png",
        ):
            (directory / name).unlink(missing_ok=True)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            "단계=%s 실험=%s 설정=%s", stage, experiment_id, metadata["config"]
        )
        logger.info(
            "장치=%s GPU=%s 저장=%s",
            device,
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "없음",
            directory,
        )
        for name in ("train", "validation", "test"):
            subset = getattr(splits, name)
            logger.info(
                "%s %d행 분포=%s",
                name,
                len(subset),
                subset.failureType.value_counts().to_dict(),
            )
        prepared_at = time.monotonic()
        logger.info("train 전처리·특징 준비 시작")
        train_data = WaferDataset(
            splits.train, config, training=True, heartbeat=status.update
        )
        logger.info("validation 전처리·특징 준비 시작")
        validation_data = WaferDataset(
            splits.validation,
            config,
            feature_stats=train_data.feature_stats,
            heartbeat=status.update,
        )
        logger.info(
            "전처리·특징 준비 완료 %.1f초", time.monotonic() - prepared_at
        )
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
        )
        model = build_model(config).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        logger.info("파라미터=%d", parameters)
        weights = None
        if config.loss == "weighted_ce":
            counts = np.bincount(splits.train.label_id, minlength=9)
            if (counts == 0).any():
                raise ValueError("train에 모든 클래스가 필요합니다.")
            weights = 1 / np.sqrt(counts)
            weights = torch.tensor(
                weights / weights.mean(), dtype=torch.float32, device=device
            )
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=config.scheduler_patience,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history, best_f1, best_epoch, waiting = [], -1.0, 0, 0
        history_details = []
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            train_loss, train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                status,
                epoch,
                optimizer,
                scaler,
            )
            val_loss, val_metrics = _run_epoch(
                model, validation_loader, criterion, device, status, epoch
            )
            train_f1 = train_metrics["macro_f1"]
            val_f1 = val_metrics["macro_f1"]
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_f1)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logger.info("학습률 변경 %.6g → %.6g", old_lr, new_lr)
            if val_f1 > best_f1:
                best_f1, best_epoch, waiting = val_f1, epoch, 0
                temporary = directory / "model.pth.tmp"
                torch.save(
                    {
                        **metadata,
                        "model_state_dict": model.state_dict(),
                        "feature_stats": train_data.feature_stats,
                        "feature_names": list(FEATURE_NAMES),
                        "best_epoch": best_epoch,
                        "best_validation_macro_f1": best_f1,
                    },
                    temporary,
                )
                atomic_replace(temporary, directory / "model.pth")
                logger.info(
                    "최고 checkpoint 저장 epoch=%d F1=%.6f", epoch, best_f1
                )
            else:
                waiting += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_macro_f1": train_f1,
                    "validation_loss": val_loss,
                    "validation_macro_f1": val_f1,
                    **_flatten_epoch_metrics("train", train_metrics),
                    **_flatten_epoch_metrics("validation", val_metrics),
                    "learning_rate": new_lr,
                    "epoch_seconds": time.monotonic() - epoch_started,
                    "best_validation_macro_f1": best_f1,
                    "early_stopping_wait": waiting,
                }
            )
            temporary = directory / "history.csv.tmp"
            pd.DataFrame(history).to_csv(temporary, index=False)
            atomic_replace(temporary, directory / "history.csv")
            history_details.append(
                {
                    "epoch": epoch,
                    "train": {"loss": train_loss, **train_metrics},
                    "validation": {"loss": val_loss, **val_metrics},
                }
            )
            write_json(
                directory / "history_details.json",
                {"classes": list(CLASSES), "epochs": history_details},
            )
            logger.info(
                "Epoch %d/%d train loss=%.4f F1=%.4f | "
                "val loss=%.4f F1=%.4f | "
                "lr=%.6g %.1f초 best=%.4f wait=%d/%d",
                epoch,
                config.epochs,
                train_loss,
                train_f1,
                val_loss,
                val_f1,
                new_lr,
                history[-1]["epoch_seconds"],
                best_f1,
                waiting,
                config.early_stopping_patience,
            )
            status.update(
                force=True,
                phase="epoch_completed",
                best_validation_macro_f1=best_f1,
                early_stopping_wait=waiting,
            )
            if waiting >= config.early_stopping_patience:
                logger.info("Early stopping: epoch %d", epoch)
                break
        status.update(force=True, phase="saving_validation")
        checkpoint = torch.load(
            directory / "model.pth", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        labels, probabilities = predict(
            model, validation_loader, device, status.update
        )
        validation = save_predictions(
            splits.validation, labels, probabilities, directory
        )
        write_json(
            directory / "metrics.json",
            {
                "validation": validation,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "elapsed_seconds": time.monotonic() - status.started,
                "torch_version": str(torch.__version__),
            },
        )
        status.update(force=True, state="completed", phase="completed")
        logger.info(
            "실험 완료: %s validation Macro-F1=%.6f",
            experiment_id,
            validation["macro_f1"],
        )
        return directory
    except (Exception, KeyboardInterrupt) as error:
        status.update(
            force=True,
            state="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "failed",
            error=str(error),
        )
        logger.exception("실험 중단/오류: %s", experiment_id)
        raise
