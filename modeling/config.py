"""재현 가능한 실험 설정과 저장 유틸리티."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CLASSES = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
)
DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/SKALA/wafer_v2")
DEFAULT_DATA_PATH = DRIVE_ROOT / "data/LSWMD.pkl"
DEFAULT_ARTIFACT_ROOT = DRIVE_ROOT / "modeling/artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    """단일 학습의 재현 설정을 보관한다.

    Attributes:
        model: 비교할 CNN 종류.
        preprocessing: fixed_resize, resize_pad 또는 resize_pad_mask.
        loss: ce 또는 weighted_ce.
        augmentation: 학습 중 온라인 증강 적용 여부.
        seed: 모델 초기화와 데이터 순서에 사용하는 시드.
        target_size: 정사각형 모델 입력의 한 변.
        epochs: 최대 학습 epoch.
        batch_size: 배치 샘플 수.
        learning_rate: Adam 초기 학습률.
        weight_decay: Adam 가중치 감쇠.
        scheduler_patience: 학습률 감소 전 대기 epoch.
        early_stopping_patience: 개선 없이 기다릴 epoch 수.
        num_workers: DataLoader 워커 수.
    """

    model: str = "small_cnn"
    preprocessing: str = "fixed_resize"
    loss: str = "weighted_ce"
    augmentation: bool = False
    seed: int = 42
    target_size: int = 64
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    early_stopping_patience: int = 7
    num_workers: int = 0

    def __post_init__(self):
        """지원하지 않는 설정을 학습 전에 거부한다."""
        choices = {
            "model": (
                "small_cnn",
                "spatial_cnn",
                "residual_cnn",
                "hybrid_cnn",
            ),
            "preprocessing": ("fixed_resize", "resize_pad", "resize_pad_mask"),
            "loss": ("ce", "weighted_ce"),
        }
        for field, allowed in choices.items():
            if getattr(self, field) not in allowed:
                raise ValueError(
                    f"지원하지 않는 {field}: {getattr(self, field)}"
                )
        if min(self.epochs, self.batch_size, self.early_stopping_patience) < 1:
            raise ValueError(
                "epoch, batch size, early stopping patience는 양수여야 합니다."
            )
        if self.target_size < 16 or self.num_workers < 0:
            raise ValueError(
                "target_size는 16 이상, num_workers는 0 이상이어야 합니다."
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate는 양수, weight_decay는 음수가 아니어야 합니다."
            )

    @property
    def in_channels(self):
        """전처리에 대응하는 입력 채널 수를 반환한다."""
        return 2 if self.preprocessing == "resize_pad_mask" else 1


def read_json(path):
    """UTF-8 JSON 파일을 읽어 반환한다.

    Args:
        path: JSON 경로.

    Returns:
        역직렬화된 JSON 값.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    """JSON을 임시 파일에 쓴 뒤 교체하여 부분 기록을 방지한다.

    Args:
        path: 출력 JSON 경로.
        payload: JSON 직렬화 가능한 값.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    atomic_replace(temporary, path)


def atomic_replace(source, destination):
    """동기화 폴더의 일시적인 파일 잠금을 짧게 재시도한다.

    Args:
        source: 기록을 마친 임시 파일 경로.
        destination: 교체할 최종 파일 경로.

    Raises:
        PermissionError: 약 3초 재시도 후에도 교체가 거부되는 경우.
    """
    for attempt, delay in enumerate((0, 0.1, 0.2, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise


def code_version():
    """패키지 Python 소스의 해시를 계산한다."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_identity(config, split_id):
    """설정·분할·코드에 종속된 실험 ID와 메타데이터를 반환한다.

    Args:
        config: ExperimentConfig 인스턴스.
        split_id: 고정 분할의 데이터 지문.

    Returns:
        실험 ID와 메타데이터 딕셔너리의 튜플.
    """
    payload = {
        "config": asdict(config),
        "split_id": split_id,
        "code_version": code_version(),
        "classes": list(CLASSES),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return f"{config.model}_{digest[:12]}", payload
