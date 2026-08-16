import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import MODEL_DIR, NEW_MODEL_HEATMAP_PATH
from config.settings import (
    BATCH_SIZE,
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABEL2ID,
    LEARNING_RATE,
    MODEL_INPUT_CHANNELS,
    MODEL_RESIZE_MODE,
    NUM_CLASSES,
    PRESENTATION_BATCH_SIZE,
    PRESENTATION_EPOCHS,
    SCHEDULER_PATIENCE,
    SEED,
    TARGET_SIZE,
    TIMEZONE,
    WEIGHT_DECAY,
)
from core.data.dataset import WaferDataset
from core.evaluation.metrics import calculate_metrics
from core.evaluation.visualization import draw_cm_heatmap
from core.model.architecture import SmallCNN
from utils.common import seed_worker, set_seed


@dataclass(frozen=True)
class DataSplits:
    """재학습에 사용하는 고정 데이터 분할입니다.

    Attributes:
        train: 전체 데이터의 80%인 학습 데이터입니다.
        validation: 전체 데이터의 10%인 검증 데이터입니다.
        test: 전체 데이터의 10%인 최종 비교 데이터입니다.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _prepare_dataframe(dataset):
    """재학습 DataFrame의 필수 컬럼과 라벨을 검증합니다."""
    required_columns = {"waferMap", "failureType", "lotName"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")

    dataframe = dataset.copy()
    dataframe["label_id"] = dataframe["failureType"].map(LABEL2ID)
    if dataframe["label_id"].isna().any():
        invalid_labels = sorted(
            dataframe.loc[
                dataframe["label_id"].isna(),
                "failureType",
            ].astype(str).unique()
        )
        raise ValueError(f"지원하지 않는 failureType입니다: {invalid_labels}")
    dataframe["label_id"] = dataframe["label_id"].astype(int)
    return dataframe.reset_index(drop=True)


def split_dataset(dataset, seed=SEED):
    """데이터를 label 계층 기반 80/10/10으로 분할합니다.

    Args:
        dataset: 웨이퍼 맵, 라벨, Lot 이름이 포함된 DataFrame입니다.
        seed: 분할 재현성에 사용할 시드입니다.

    Returns:
        학습, 검증, 테스트 DataFrame을 담은 :class:`DataSplits`입니다.
    """
    dataframe = _prepare_dataframe(dataset)
    train_frame, temporary_frame = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=seed,
        stratify=dataframe["label_id"],
    )
    validation_frame, test_frame = train_test_split(
        temporary_frame,
        test_size=0.5,
        random_state=seed,
        stratify=temporary_frame["label_id"],
    )
    return DataSplits(
        train=train_frame.reset_index(drop=True),
        validation=validation_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )


def _oversample_training_frame(dataframe):
    """각 클래스 크기를 원본 클래스 평균 크기로 맞춥니다."""
    average_size = int(dataframe["label_id"].value_counts().mean())
    sampled_frames = []
    for label in sorted(dataframe["label_id"].unique()):
        class_frame = dataframe[dataframe["label_id"] == label]
        sampled_frames.append(
            resample(
                class_frame,
                replace=True,
                n_samples=average_size,
                random_state=SEED,
            )
        )
    return (
        pd.concat(sampled_frames)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def _build_data_loaders(data_splits, batch_size):
    """exp5 전처리와 증강을 사용하는 DataLoader를 만듭니다."""
    training_frame = _oversample_training_frame(data_splits.train)
    training_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
        ]
    )

    def build_dataset(dataframe, augmentation=None):
        return WaferDataset(
            dataframe["waferMap"].values,
            dataframe["label_id"].values,
            dataframe["lotName"].values,
            transforms=augmentation,
            resize_mode=MODEL_RESIZE_MODE,
            target_size=TARGET_SIZE,
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)
    common_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        build_dataset(training_frame, training_transforms),
        shuffle=True,
        generator=generator,
        **common_options,
    )
    validation_loader = DataLoader(
        build_dataset(data_splits.validation),
        shuffle=False,
        **common_options,
    )
    test_loader = DataLoader(
        build_dataset(data_splits.test),
        shuffle=False,
        **common_options,
    )
    return train_loader, validation_loader, test_loader


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    description="Validation",
):
    """학습 또는 평가 epoch을 실행합니다."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_sum = 0.0
    total = 0
    all_predictions = []
    all_labels = []
    all_lots = []
    use_amp = device.type == "cuda"

    for inputs, labels, lot_names in tqdm(loader, desc=description):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        gradient_context = torch.enable_grad() if is_training else torch.no_grad()
        with gradient_context:
            with amp_context:
                logits = model(inputs)
                loss = criterion(logits, labels)

        if is_training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        loss_sum += loss.item() * inputs.size(0)
        total += inputs.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_lots.extend(list(lot_names))

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    return (
        loss_sum / total,
        macro_f1,
        all_labels,
        all_predictions,
        all_lots,
    )


def _checkpoint_config(batch_size, epochs):
    """재학습 체크포인트에 기록할 모델 설정을 반환합니다."""
    return {
        "checkpoint_version": 1,
        "name": "exp5_resize_pad_mask_os_aug_retrained",
        "resize_mode": MODEL_RESIZE_MODE,
        "use_oversampling": True,
        "use_augmentation": True,
        "in_channels": MODEL_INPUT_CHANNELS,
        "classes": list(CLASSES),
        "batch_size": batch_size,
        "target_size": list(TARGET_SIZE),
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }


def process(dataset, is_presentation, data_splits=None):
    """exp5 구조의 후보 모델을 새 초기값부터 학습합니다.

    Args:
        dataset: 전체 업로드 DataFrame입니다.
        is_presentation: 발표용 단축 학습 사용 여부입니다.
        data_splits: 비교에 재사용할 선택적 고정 분할입니다.

    Returns:
        기존 API가 사용하는 성능, 예측, Lot, 체크포인트 경로입니다.
    """
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if data_splits is None:
        data_splits = split_dataset(dataset)

    batch_size = PRESENTATION_BATCH_SIZE if is_presentation else BATCH_SIZE
    epochs = PRESENTATION_EPOCHS if is_presentation else EPOCHS
    train_loader, validation_loader, test_loader = _build_data_loaders(
        data_splits,
        batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(
        num_classes=NUM_CLASSES,
        in_channels=MODEL_INPUT_CHANNELS,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda"
        else None
    )

    current_time = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    model_location = os.path.join(MODEL_DIR, f"{current_time}_model.pth")
    os.makedirs(os.path.dirname(model_location), exist_ok=True)
    checkpoint_config = _checkpoint_config(batch_size, epochs)
    best_validation_f1 = -np.inf
    best_model_state = copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_f1, _, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            description="Train",
        )
        validation_loss, validation_f1, _, _, _ = _run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            description="Validation",
        )
        scheduler.step(validation_f1)
        print(
            f"[Epoch {epoch:03d}] "
            f"train loss {train_loss:.4f} f1 {train_f1:.4f} | "
            f"validation loss {validation_loss:.4f} "
            f"f1 {validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "config": checkpoint_config,
                    "best_val_f1": best_validation_f1,
                },
                model_location,
            )
        else:
            patience_count += 1
            if patience_count >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_model_state)
    model.eval()
    _, _, all_labels, all_predictions, all_lots = _run_epoch(
        model,
        test_loader,
        criterion,
        device,
        description="Test",
    )
    (
        test_accuracy,
        test_macro_f1,
        test_macro_precision,
        test_macro_recall,
        metrics_by_category,
    ) = calculate_metrics(all_labels, all_predictions)

    try:
        draw_cm_heatmap(
            all_labels,
            all_predictions,
            NEW_MODEL_HEATMAP_PATH,
            palette="Peach",
            normalize="true",
        )
    except Exception as error:
        print(f"Confusion Matrix 생성을 건너뜁니다: {error}")

    return (
        test_accuracy,
        test_macro_f1,
        test_macro_precision,
        test_macro_recall,
        metrics_by_category,
        all_labels,
        all_predictions,
        all_lots,
        model_location,
    )
