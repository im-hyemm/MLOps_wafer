import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

BACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK_DIR))

from config.settings import CLASSES  # noqa: E402
from core.model.architecture import load_model  # noqa: E402
from core.model.predictor import (  # noqa: E402
    process_labeled_images,
    process_multi_images,
    process_multi_images_one_lot,
    process_one_image,
)
from utils.image_utils import (  # noqa: E402
    preprocess_wafer_map,
    resize_pad_with_mask,
)


class ModelPipelineTest(unittest.TestCase):
    """legacy와 exp5 모델 파이프라인을 검증합니다."""

    @classmethod
    def setUpClass(cls):
        """실제 체크포인트 경로와 CPU 장치를 준비합니다."""
        cls.device = torch.device("cpu")
        cls.legacy_path = BACK_DIR / "model" / "best_model_legacy.pth"
        cls.presentation_path = BACK_DIR / "model" / "presentation_model.pth"
        cls.exp5_path = BACK_DIR / "model" / "best_model.pth"

    def test_resize_pad_mask_matches_exp5_shape(self):
        """비정사각형 이미지가 올바른 크기와 mask로 변환됩니다."""
        image = np.ones((3, 5), dtype=np.uint8)
        padded, mask = resize_pad_with_mask(image, (8, 8))

        self.assertEqual(padded.shape, (8, 8))
        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(set(np.unique(mask)), {0, 1})
        self.assertEqual(int(mask.sum()), 40)

    def test_exp5_preprocessing_builds_two_channels(self):
        """exp5 전처리가 이미지와 mask의 두 채널을 생성합니다."""
        image = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        array = preprocess_wafer_map(
            image,
            resize_mode="resize_pad_mask",
            target_size=(64, 64),
        )

        self.assertEqual(array.shape, (2, 64, 64))
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(np.isin(array[0], [0.0, 0.5, 1.0]).all())
        self.assertTrue(np.isin(array[1], [0.0, 1.0]).all())

    def test_legacy_keeps_truncating_resize_behavior(self):
        """legacy 전처리는 기존 정수 절삭 리사이즈를 유지합니다."""
        image = np.ones((3, 5), dtype=np.uint8)
        legacy = preprocess_wafer_map(
            image,
            resize_mode="resize_pad",
            target_size=(8, 8),
        )
        exp5 = preprocess_wafer_map(
            image,
            resize_mode="resize_pad_mask",
            target_size=(8, 8),
        )

        self.assertEqual(int(np.count_nonzero(legacy[0])), 32)
        self.assertEqual(int(np.count_nonzero(exp5[0])), 40)

    def test_exp5_checkpoint_produces_nine_logits(self):
        """exp5 체크포인트가 2채널 입력에서 9개 logit을 생성합니다."""
        model, spec = load_model(self.exp5_path, self.device)
        outputs = model(torch.zeros((2, 2, 64, 64)))

        self.assertEqual(spec.in_channels, 2)
        self.assertEqual(spec.resize_mode, "resize_pad_mask")
        self.assertEqual(spec.classes, tuple(CLASSES))
        self.assertEqual(tuple(outputs.shape), (2, 9))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_legacy_checkpoints_remain_loadable(self):
        """백업 및 발표용 체크포인트가 1채널로 계속 동작합니다."""
        for path in (self.legacy_path, self.presentation_path):
            with self.subTest(path=path.name):
                model, spec = load_model(path, self.device)
                outputs = model(torch.zeros((1, 1, 64, 64)))

                self.assertEqual(spec.in_channels, 1)
                self.assertEqual(spec.resize_mode, "resize_pad")
                self.assertEqual(tuple(outputs.shape), (1, 9))

    @staticmethod
    def _image_buffer(value):
        """테스트용 grayscale PNG 버퍼를 만듭니다."""
        image = np.full((24, 32), value, dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def test_single_image_prediction_contract(self):
        """단일 이미지 추론 반환 계약을 유지합니다."""
        is_defect, label_id, confidence = process_one_image(
            self._image_buffer(1)
        )

        self.assertIsInstance(is_defect, bool)
        self.assertIn(label_id, range(len(CLASSES)))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_zip_prediction_contracts(self):
        """단일 Lot과 다중 Lot ZIP 추론 계약을 유지합니다."""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as archive:
            for index, value in enumerate((1, 2), start=1):
                archive.writestr(
                    f"lotA_{index}.png",
                    self._image_buffer(value).getvalue(),
                )

        zip_buffer.seek(0)
        one_lot_result = process_multi_images_one_lot(zip_buffer)
        self.assertEqual(one_lot_result[0], "lotA")
        self.assertEqual(one_lot_result[2], 2)
        self.assertEqual(len(one_lot_result[6]), 2)

        zip_buffer.seek(0)
        multi_lot_result = process_multi_images(zip_buffer)
        self.assertEqual(multi_lot_result[1], 2)
        self.assertEqual(len(multi_lot_result[5]), 2)

    def test_labeled_prediction_contract(self):
        """라벨 DataFrame 평가의 기존 반환 계약을 유지합니다."""
        dataframe = pd.DataFrame(
            {
                "waferMap": [
                    np.zeros((20, 30), dtype=np.uint8),
                    np.ones((25, 25), dtype=np.uint8),
                ],
                "failureType": ["none", "none"],
                "lotName": ["lotA", "lotA"],
            }
        )
        result = process_labeled_images(
            dataframe,
            self.exp5_path,
            batch_size=2,
            num_workers=0,
        )

        self.assertEqual(len(result), 8)
        self.assertEqual(result[5], [8, 8])
        self.assertEqual(len(result[6]), 2)
        self.assertEqual(result[7], ["lotA", "lotA"])


if __name__ == "__main__":
    unittest.main()
