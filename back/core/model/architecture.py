from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config.settings import CLASSES, NUM_CLASSES, TARGET_SIZE


@dataclass(frozen=True)
class ModelSpec:
    """모델 입력과 클래스 계약을 나타냅니다.

    Attributes:
        in_channels: 모델 입력 채널 수입니다.
        resize_mode: 추론 전처리 방식입니다.
        target_size: 입력 이미지의 목표 높이와 너비입니다.
        classes: 출력 인덱스 순서에 대응하는 클래스 이름입니다.
    """

    in_channels: int
    resize_mode: str
    target_size: tuple[int, int]
    classes: tuple[str, ...]


class SmallCNN(nn.Module):
    """웨이퍼 맵 분류용 CNN 모델입니다."""

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        in_channels=1,
    ):
        """모델을 초기화합니다.

        Args:
            num_classes: 출력 클래스 수입니다.
            in_channels: 입력 이미지 채널 수입니다.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """입력 텐서의 클래스별 logit을 계산합니다.

        Args:
            x: ``(배치, 채널, 높이, 너비)`` 형태의 입력입니다.

        Returns:
            ``(배치, 클래스)`` 형태의 logit입니다.
        """
        x = self.features(x)
        return self.classifier(x)


def _normalize_target_size(value: Any) -> tuple[int, int]:
    """체크포인트 입력 크기를 높이와 너비 튜플로 정규화합니다."""
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"올바르지 않은 target_size입니다: {value}")


def _unpack_checkpoint(checkpoint):
    """체크포인트에서 state dict와 설정을 분리합니다."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    return checkpoint, {}


def _build_model_spec(state_dict, config):
    """가중치와 설정을 검증해 모델 사양을 생성합니다."""
    first_weight = state_dict.get("features.0.weight")
    output_weight = state_dict.get("classifier.2.weight")
    if first_weight is None or output_weight is None:
        raise ValueError("지원하지 않는 SmallCNN 체크포인트입니다.")

    in_channels = int(first_weight.shape[1])
    num_classes = int(output_weight.shape[0])
    if in_channels not in (1, 2):
        raise ValueError(f"지원하지 않는 입력 채널 수입니다: {in_channels}")
    configured_channels = int(config.get("in_channels", in_channels))
    if configured_channels != in_channels:
        raise ValueError(
            "체크포인트 입력 채널과 config의 in_channels가 다릅니다."
        )
    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"체크포인트 클래스 수가 {NUM_CLASSES}개가 아닙니다: "
            f"{num_classes}"
        )

    default_resize_mode = (
        "resize_pad_mask" if in_channels == 2 else "resize_pad"
    )
    resize_mode = config.get("resize_mode", default_resize_mode)
    expected_channels = 2 if resize_mode == "resize_pad_mask" else 1
    if resize_mode not in ("resize_pad", "resize_pad_mask"):
        raise ValueError(f"지원하지 않는 resize_mode입니다: {resize_mode}")
    if in_channels != expected_channels:
        raise ValueError(
            "체크포인트 입력 채널과 전처리 설정이 일치하지 않습니다."
        )

    classes = tuple(config.get("classes", CLASSES))
    if classes != tuple(CLASSES):
        raise ValueError("체크포인트 클래스 순서가 서비스 설정과 다릅니다.")

    return ModelSpec(
        in_channels=in_channels,
        resize_mode=resize_mode,
        target_size=_normalize_target_size(
            config.get("target_size", TARGET_SIZE)
        ),
        classes=classes,
    )


def load_model(model_path, device):
    """체크포인트를 읽어 모델과 입력 사양을 반환합니다.

    Args:
        model_path: raw state dict 또는 bundle 체크포인트 경로입니다.
        device: 모델을 배치할 PyTorch 장치입니다.

    Returns:
        평가 모드 모델과 :class:`ModelSpec`의 튜플입니다.
    """
    checkpoint = torch.load(
        Path(model_path),
        map_location=device,
        weights_only=True,
    )
    state_dict, config = _unpack_checkpoint(checkpoint)
    spec = _build_model_spec(state_dict, config)

    model = SmallCNN(
        num_classes=len(spec.classes),
        in_channels=spec.in_channels,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec
