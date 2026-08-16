import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

BACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK_DIR))

from api.routes.training import (  # noqa: E402
    calculate_raw_macro_f1,
    promote_checkpoint,
    should_promote_model,
)
from config.settings import RETRAIN_F1_THRESHOLD  # noqa: E402
from core.model.trainer import split_dataset  # noqa: E402


class RetrainingPipelineTest(unittest.TestCase):
    """재학습 분할과 모델 승격 정책을 검증합니다."""

    @staticmethod
    def _build_dataset():
        """각 클래스가 20개씩 포함된 테스트 DataFrame을 만듭니다."""
        class_names = [
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Random",
            "Scratch",
            "Near-full",
            "none",
        ]
        rows = []
        for label in class_names:
            for index in range(20):
                rows.append(
                    {
                        "waferMap": np.zeros((4, 5), dtype=np.uint8),
                        "failureType": label,
                        "lotName": f"{label}-{index}",
                    }
                )
        return pd.DataFrame(rows)

    def test_threshold_is_preserved(self):
        """재학습 추천 임계치가 0.7로 유지됩니다."""
        self.assertEqual(RETRAIN_F1_THRESHOLD, 0.7)
        self.assertTrue(0.699 < RETRAIN_F1_THRESHOLD)
        self.assertFalse(0.7 < RETRAIN_F1_THRESHOLD)

    def test_split_is_reproducible_80_10_10(self):
        """동일 seed에서 80/10/10 계층 분할이 재현됩니다."""
        dataset = self._build_dataset()
        first = split_dataset(dataset)
        second = split_dataset(dataset)

        self.assertEqual(len(first.train), 144)
        self.assertEqual(len(first.validation), 18)
        self.assertEqual(len(first.test), 18)
        self.assertEqual(
            first.test["lotName"].tolist(),
            second.test["lotName"].tolist(),
        )
        self.assertTrue(
            (first.test["label_id"].value_counts().sort_index() == 2).all()
        )

    def test_raw_macro_f1_uses_fixed_nine_classes(self):
        """승격 점수가 9개 고정 클래스 기준으로 계산됩니다."""
        score = calculate_raw_macro_f1([0, 1], [0, 1])
        self.assertAlmostEqual(score, 2 / 9)

    def test_candidate_must_be_strictly_better(self):
        """동점 후보는 승격되지 않고 높은 후보만 승격됩니다."""
        self.assertFalse(should_promote_model(0.8, 0.8))
        self.assertFalse(should_promote_model(0.8, 0.79))
        self.assertTrue(should_promote_model(0.8, 0.81))

    def test_checkpoint_promotion_replaces_target(self):
        """후보 체크포인트가 대상 파일에 원자적으로 반영됩니다."""
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.pth"
            target = Path(directory) / "best_model.pth"
            candidate.write_bytes(b"candidate")
            target.write_bytes(b"incumbent")

            promote_checkpoint(candidate, target)

            self.assertEqual(target.read_bytes(), b"candidate")
            self.assertFalse(Path(f"{target}.tmp").exists())

    def test_copy_failure_preserves_target(self):
        """후보 복사 실패 시 현재 대상 파일을 보존합니다."""
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.pth"
            target = Path(directory) / "best_model.pth"
            candidate.write_bytes(b"candidate")
            target.write_bytes(b"incumbent")

            with patch("shutil.copyfile", side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    promote_checkpoint(candidate, target)

            self.assertEqual(target.read_bytes(), b"incumbent")
            self.assertFalse(os.path.exists(f"{target}.tmp"))


if __name__ == "__main__":
    unittest.main()
