import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import TARGET_SIZE
from utils.image_utils import preprocess_wafer_map


class WaferDataset(Dataset):
    """학습과 평가에 사용하는 웨이퍼 데이터셋입니다."""

    def __init__(
        self,
        images,
        labels,
        lot_names,
        resize_mode="resize_pad",
        target_size=TARGET_SIZE,
    ):
        """데이터셋을 초기화합니다.

        Args:
            images: 웨이퍼 맵 배열 모음입니다.
            labels: 클래스 ID 모음입니다.
            lot_names: Lot 이름 모음입니다.
            resize_mode: 모델 입력 전처리 방식입니다.
            target_size: 입력 이미지의 목표 높이와 너비입니다.
        """
        self.images = images
        self.labels = labels
        self.lot_names = lot_names
        self.resize_mode = resize_mode
        self.target_size = target_size

    def __len__(self):
        """데이터 개수를 반환합니다."""
        return len(self.labels)

    def __getitem__(self, idx):
        """전처리한 웨이퍼와 라벨, Lot 이름을 반환합니다."""
        image = np.asarray(self.images[idx], dtype=np.uint8)
        array = preprocess_wafer_map(
            image,
            resize_mode=self.resize_mode,
            target_size=self.target_size,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(array))

        return tensor, int(self.labels[idx]), self.lot_names[idx]


class WaferInferenceDataset(Dataset):
    """라벨 데이터 평가에 사용하는 웨이퍼 데이터셋입니다."""

    def __init__(
        self,
        wafer_series,
        resize_mode="resize_pad",
        target_size=TARGET_SIZE,
    ):
        """추론 데이터셋을 초기화합니다.

        Args:
            wafer_series: 웨이퍼 맵 pandas Series입니다.
            resize_mode: 모델 입력 전처리 방식입니다.
            target_size: 입력 이미지의 목표 높이와 너비입니다.
        """
        self.wafer_series = wafer_series.reset_index(drop=True)
        self.resize_mode = resize_mode
        self.target_size = target_size

    def __len__(self):
        """데이터 개수를 반환합니다."""
        return len(self.wafer_series)

    def __getitem__(self, idx):
        """전처리한 모델 입력 텐서를 반환합니다."""
        image = np.asarray(self.wafer_series.iloc[idx], dtype=np.uint8)
        array = preprocess_wafer_map(
            image,
            resize_mode=self.resize_mode,
            target_size=self.target_size,
        )
        return torch.from_numpy(np.ascontiguousarray(array))
