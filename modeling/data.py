"""데이터 정리, 고정 분할과 학습용 Dataset."""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import CLASSES, read_json, write_json
from .features import extract_geometric_features
from .preprocessing import augment_map, encode_map, preprocess_map

LOGGER = logging.getLogger("modeling")


@dataclass
class DataSplits:
    """모든 실험이 공유하는 고정 분할과 식별자.

    Attributes:
        train: 전체의 80% 학습 데이터.
        validation: 전체의 10% 후보 선택 데이터.
        test: 전체의 10% 최종 평가 데이터.
        split_id: 데이터 내용과 분할 정책을 포함한 지문.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_id: str


def _unwrap(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if np.size(value) == 0:
            return None
        value = np.asarray(value, dtype=object).reshape(-1)[0]
    return value


def load_dataset(data_path):
    """중첩 라벨을 문자열로 바꾸고 라벨이 있는 행만 반환한다.

    Args:
        data_path: 신뢰할 수 있는 LSWMD pickle 파일 경로.

    Returns:
        원본 행 번호와 9개 클래스 label_id가 포함된 DataFrame.
    """
    started = time.monotonic()
    LOGGER.info("데이터 로딩 시작: %s", data_path)
    frame = pd.read_pickle(data_path).rename(
        columns={"trianTestLabel": "trainTestLabel"}
    )
    required = {"waferMap", "failureType", "lotName"}
    if not required.issubset(frame.columns):
        raise ValueError(f"필수 컬럼 누락: {required - set(frame.columns)}")
    frame = frame.copy()
    frame["source_index"] = np.arange(len(frame))
    frame["failureType"] = frame["failureType"].map(_unwrap)
    labeled = frame["failureType"].notna() & frame["failureType"].ne("")
    invalid = set(frame.loc[labeled, "failureType"]) - set(CLASSES)
    if invalid:
        raise ValueError(f"지원하지 않는 라벨: {invalid}")
    frame = frame.loc[labeled].copy().reset_index(drop=True)
    frame["label_id"] = (
        frame["failureType"].map(dict(zip(CLASSES, range(9)))).astype(int)
    )
    frame["lotName"] = frame["lotName"].astype(str)
    LOGGER.info(
        "데이터 로딩 완료: %s행, %.1f초",
        len(frame),
        time.monotonic() - started,
    )
    return frame


def create_or_load_splits(frame, artifact_root):
    """데이터 해시로 검증한 셔플·층화 80/10/10 분할을 저장·재사용한다.

    Args:
        frame: load_dataset으로 정리한 DataFrame.
        artifact_root: 분할 manifest를 저장할 루트.

    Returns:
        고정 분할과 split_id를 포함한 DataSplits.

    Raises:
        ValueError: manifest가 다르거나 분할에 클래스가 빠진 경우.
    """
    LOGGER.info("데이터 지문 및 고정 분할 준비 시작")
    digest = hashlib.sha256(b"stratified-80-10-10-seed42-v1")
    for row in tqdm(
        frame.itertuples(), total=len(frame), desc="분할: 데이터 지문"
    ):
        array = np.asarray(row.waferMap)
        digest.update(
            str(
                (
                    row.source_index,
                    row.label_id,
                    row.lotName,
                    array.shape,
                    str(array.dtype),
                )
            ).encode()
        )
        digest.update(np.ascontiguousarray(array).tobytes())
    split_id = digest.hexdigest()
    path = Path(artifact_root) / "splits" / f"{split_id}.json"
    if path.exists():
        manifest = read_json(path)
        LOGGER.info("고정 분할 재사용: %s", path)
    else:
        train, temporary = train_test_split(
            frame.source_index,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=frame.label_id,
        )
        indexed = frame.set_index("source_index")
        validation, test = train_test_split(
            temporary,
            test_size=0.5,
            random_state=42,
            shuffle=True,
            stratify=indexed.loc[temporary, "label_id"],
        )
        manifest = {
            "split_id": split_id,
            "seed": 42,
            "train": train.tolist(),
            "validation": validation.tolist(),
            "test": test.tolist(),
        }
        write_json(path, manifest)
    groups = [manifest[name] for name in ("train", "validation", "test")]
    flattened = [index for group in groups for index in group]
    if manifest["split_id"] != split_id or len(flattened) != len(
        set(flattened)
    ):
        raise ValueError("분할 manifest 식별자 또는 인덱스 중복 오류")
    if set(flattened) != set(frame.source_index):
        raise ValueError("분할 manifest와 데이터 행이 다릅니다.")
    indexed = frame.set_index("source_index", drop=False)
    frames = [indexed.loc[group].reset_index(drop=True) for group in groups]
    for name, subset in zip(("train", "validation", "test"), frames):
        if set(subset.label_id) != set(range(9)):
            raise ValueError(f"{name}에 9개 클래스가 모두 있어야 합니다.")
        LOGGER.info(
            "%s: %d행, 분포=%s",
            name,
            len(subset),
            subset.failureType.value_counts().to_dict(),
        )
    lot_overlap = len(set(frames[0].lotName) & set(frames[1].lotName))
    LOGGER.info(
        "분할 완료: %s, train/validation 중복 Lot 수=%d",
        split_id[:12],
        lot_overlap,
    )
    return DataSplits(*frames, split_id)


class WaferDataset(Dataset):
    """범주 맵을 미리 준비하고 학습 시 온라인 증강을 적용한다."""

    def __init__(
        self, frame, config, training=False, feature_stats=None, heartbeat=None
    ):
        """분할별 맵과 필요 시 특징을 준비한다.

        Args:
            frame: 하나의 분할 DataFrame.
            config: 전처리와 모델 설정.
            training: train 통계 계산 및 증강 활성 여부.
            feature_stats: train에서 구한 특징 mean/std.
            heartbeat: 준비 중 상태를 갱신할 콜백.
        """
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.training = training
        self.maps = []
        raw_features = []
        for index, wafer_map in enumerate(
            tqdm(frame.waferMap, desc="전처리·특징 준비")
        ):
            prepared = preprocess_map(
                wafer_map, config.preprocessing, config.target_size
            )
            self.maps.append(prepared)
            if config.model == "hybrid_cnn":
                raw_features.append(extract_geometric_features(prepared))
            if heartbeat:
                heartbeat(
                    phase="preparing",
                    batch=index + 1,
                    total_batches=len(frame),
                )
        self.feature_stats = feature_stats
        self.raw_features = None
        if raw_features:
            self.raw_features = np.stack(raw_features)
            if training and feature_stats is None:
                deviation = self.raw_features.std(axis=0)
                # 상수 특징에 증강으로 변화가 생겨도 값이 폭증하지 않게 한다.
                self.feature_stats = {
                    "mean": self.raw_features.mean(axis=0).tolist(),
                    "std": np.where(deviation < 1e-6, 1.0, deviation).tolist(),
                }
            if self.feature_stats is None:
                raise ValueError(
                    "평가용 Hybrid 특징에는 train 정규화 통계가 필요합니다."
                )

    def __len__(self):
        """샘플 개수를 반환한다."""
        return len(self.frame)

    def __getitem__(self, index):
        """동일 맵에서 이미지와 특징을 생성하고 정답을 반환한다.

        Args:
            index: 샘플 위치.

        Returns:
            이미지 tensor, 특징 tensor, label_id 튜플.
        """
        wafer_map = self.maps[index]
        augmented = self.training and self.config.augmentation
        if augmented:
            wafer_map = augment_map(wafer_map)
        features = np.empty(0, dtype=np.float32)
        if self.raw_features is not None:
            raw = (
                extract_geometric_features(wafer_map)
                if augmented
                else self.raw_features[index]
            )
            features = (
                (raw - np.asarray(self.feature_stats["mean"]))
                / np.asarray(self.feature_stats["std"])
            ).astype(np.float32)
        return (
            torch.from_numpy(encode_map(wafer_map, self.config.preprocessing)),
            torch.from_numpy(features),
            int(self.frame.iloc[index].label_id),
        )
