"""웨이퍼 분류용 네 가지 작은 CNN 아키텍처."""

import torch
from torch import nn


def _conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _backbone(in_channels, pool_size):
    layers = []
    for index, channels in enumerate((32, 64, 128, 256)):
        layers.extend(list(_conv(in_channels, channels).children()))
        if index < 3:
            layers.append(nn.MaxPool2d(2))
        in_channels = channels
    layers.append(nn.AdaptiveAvgPool2d(pool_size))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """기존 4블록·1×1 pooling 기준 모델."""

    def __init__(self, in_channels=1):
        """기존 서비스와 동일한 특징 추출기와 분류기를 생성한다."""
        super().__init__()
        self.features = _backbone(in_channels, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 9)
        )

    def forward(self, images, features=None):
        """이미지로부터 9개 클래스 logits를 반환한다."""
        return self.classifier(self.features(images))


class SpatialCNN(nn.Module):
    """2×2 pooling과 128차원 표현을 사용하는 CNN."""

    def __init__(self, in_channels=1):
        """공간 특징과 작은 MLP 분류기를 생성한다."""
        super().__init__()
        self.features = _backbone(in_channels, 2)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 9))

    def forward(self, images, features=None):
        """공간 특징으로 9개 클래스 logits를 반환한다."""
        return self.classifier(self.embedding(self.features(images)))


class ResidualBlock(nn.Module):
    """동일 채널의 두 Conv와 잔차 연결을 사용하는 블록."""

    def __init__(self, channels):
        """두 Conv의 변환 경로를 생성한다."""
        super().__init__()
        self.layers = nn.Sequential(
            _conv(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs):
        """원본 입력과 변환 결과를 더해 반환한다."""
        return torch.relu(inputs + self.layers(inputs))


class ResidualCNN(nn.Module):
    """32→64→128 채널 잔차 CNN."""

    def __init__(self, in_channels=1):
        """세 잔차 블록과 공간 분류기를 생성한다."""
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
            nn.Linear(128, 9),
        )

    def forward(self, images, features=None):
        """잔차 특징으로 9개 클래스 logits를 반환한다."""
        return self.classifier(self.features(images))


class HybridCNN(nn.Module):
    """SpatialCNN 128차원 표현과 24개 기하학 특징을 결합한다."""

    def __init__(self, in_channels=1):
        """이미지·특징 경로와 공동 분류기를 생성한다."""
        super().__init__()
        self.features = _backbone(in_channels, 2)
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(1024, 128), nn.ReLU()
        )
        self.tabular = nn.Sequential(nn.Linear(24, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(160, 9))

    def forward(self, images, features):
        """이미지 표현과 특징 표현을 결합해 logits를 반환한다."""
        embedding = self.embedding(self.features(images))
        return self.classifier(
            torch.cat((embedding, self.tabular(features)), dim=1)
        )


def build_model(config):
    """실험 설정에 대응하는 모델을 생성한다."""
    models = {
        "small_cnn": SmallCNN,
        "spatial_cnn": SpatialCNN,
        "residual_cnn": ResidualCNN,
        "hybrid_cnn": HybridCNN,
    }
    return models[config.model](config.in_channels)
