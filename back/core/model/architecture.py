"""ResidualCNN 모델 구조와 체크포인트 로더."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config.settings import CLASSES, NUM_CLASSES, TARGET_SIZE


@dataclass(frozen=True)
class ModelSpec:
    """모델 입력과 클래스 계약을 나타냅니다.

    Attributes:
        model_name: 체크포인트에 기록된 모델 이름입니다.
        in_channels: 모델 입력 채널 수입니다.
        resize_mode: 추론 전처리 방식입니다.
        target_size: 입력 이미지의 목표 높이와 너비입니다.
        classes: 출력 인덱스 순서에 대응하는 클래스 이름입니다.
    """

    model_name: str
    in_channels: int
    resize_mode: str
    target_size: tuple[int, int]
    classes: tuple[str, ...]


def _conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """합성곱, 배치 정규화, 활성화 레이어를 생성합니다.

    Args:
        in_channels: 입력 채널 수입니다.
        out_channels: 출력 채널 수입니다.

    Returns:
        순서대로 연결된 합성곱 블록입니다.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    """동일 채널의 두 Conv와 잔차 연결을 사용하는 블록입니다."""

    def __init__(self, channels: int):
        """두 Conv의 변환 경로를 생성합니다.

        Args:
            channels: 입력과 출력에 공통으로 사용할 채널 수입니다.
        """
        super().__init__()
        self.layers = nn.Sequential(
            _conv(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """입력과 변환 결과를 더해 활성화합니다.

        Args:
            inputs: 잔차 블록 입력 텐서입니다.

        Returns:
            잔차 연결과 ReLU를 적용한 텐서입니다.
        """
        return torch.relu(inputs + self.layers(inputs))


class ResidualCNN(nn.Module):
    """웨이퍼 맵 분류용 32→64→128 채널 잔차 CNN입니다."""

    def __init__(self, in_channels: int = 1):
        """잔차 특징 추출기와 9클래스 분류기를 생성합니다.

        Args:
            in_channels: 모델 입력 채널 수입니다.
        """
        super().__init__()
        layers = []
        for channels in (32, 64, 128):
            layers.extend(
                [
                    _conv(in_channels, channels),
                    ResidualBlock(channels),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """입력 텐서의 클래스별 logit을 계산합니다.

        Args:
            inputs: ``(배치, 1, 높이, 너비)`` 형태의 입력입니다.

        Returns:
            ``(배치, 9)`` 형태의 logit입니다.
        """
        return self.classifier(self.features(inputs))


def _normalize_target_size(value: Any) -> tuple[int, int]:
    """체크포인트 입력 크기를 높이와 너비 튜플로 정규화합니다.

    Args:
        value: 정수 또는 높이와 너비의 2개 값입니다.

    Returns:
        정규화된 높이와 너비 튜플입니다.

    Raises:
        ValueError: 지원하지 않는 크기 형식인 경우 발생합니다.
    """
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"올바르지 않은 target_size입니다: {value}")


def _build_model_spec(checkpoint: dict) -> ModelSpec:
    """체크포인트 메타데이터와 가중치를 검증합니다.

    Args:
        checkpoint: 모델링 파이프라인이 저장한 체크포인트입니다.

    Returns:
        검증된 ResidualCNN 입력 사양입니다.

    Raises:
        ValueError: 체크포인트 계약이 배포 모델과 다른 경우 발생합니다.
    """
    if "model_state_dict" not in checkpoint:
        raise ValueError("model_state_dict가 없는 체크포인트입니다.")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("config가 없는 체크포인트입니다.")

    config = checkpoint["config"]
    if config.get("model") != "residual_cnn":
        raise ValueError("ResidualCNN 체크포인트만 지원합니다.")
    if config.get("preprocessing") != "resize_pad":
        raise ValueError("resize_pad 전처리 체크포인트만 지원합니다.")

    target_size = _normalize_target_size(config.get("target_size"))
    if target_size != tuple(TARGET_SIZE):
        raise ValueError(
            f"체크포인트 입력 크기가 {tuple(TARGET_SIZE)}가 아닙니다."
        )

    classes = tuple(checkpoint.get("classes", ()))
    if classes != tuple(CLASSES):
        raise ValueError("체크포인트 클래스 순서가 서비스 설정과 다릅니다.")

    state_dict = checkpoint["model_state_dict"]
    first_weight = state_dict.get("features.0.0.weight")
    output_weight = state_dict.get("classifier.5.weight")
    if first_weight is None or output_weight is None:
        raise ValueError("ResidualCNN 가중치 구조가 올바르지 않습니다.")
    if int(first_weight.shape[1]) != 1:
        raise ValueError("ResidualCNN 입력 채널 수가 1이 아닙니다.")
    if int(output_weight.shape[0]) != NUM_CLASSES:
        raise ValueError(
            f"체크포인트 클래스 수가 {NUM_CLASSES}개가 아닙니다."
        )

    return ModelSpec(
        model_name="residual_cnn",
        in_channels=1,
        resize_mode="resize_pad",
        target_size=target_size,
        classes=classes,
    )


def load_model(model_path: str | Path, device: torch.device):
    """ResidualCNN 체크포인트를 읽어 모델과 입력 사양을 반환합니다.

    Args:
        model_path: 모델링 bundle 체크포인트 경로입니다.
        device: 모델을 배치할 PyTorch 장치입니다.

    Returns:
        평가 모드 ResidualCNN과 :class:`ModelSpec`의 튜플입니다.
    """
    checkpoint = torch.load(
        Path(model_path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("bundle 형식의 ResidualCNN 체크포인트가 아닙니다.")
    spec = _build_model_spec(checkpoint)

    model = ResidualCNN(in_channels=spec.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spec
