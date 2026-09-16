import copy
import sys
import tempfile
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

from config.settings import CLASSES, TARGET_SIZE  # noqa: E402
from core.model.architecture import (  # noqa: E402
    ResidualCNN,
    load_model,
)
from core.model.predictor import (  # noqa: E402
    process_labeled_images,
    process_multi_images,
    process_multi_images_one_lot,
    process_one_image,
)
from utils.image_utils import (  # noqa: E402
    preprocess_wafer_map,
    resize_and_pad,
)


class ModelPipelineTest(unittest.TestCase):
    """ResidualCNN 전용 백엔드 모델 파이프라인을 검증합니다."""

    @classmethod
    def setUpClass(cls):
        """실제 배포 체크포인트 경로와 CPU 장치를 준비합니다."""
        cls.device = torch.device("cpu")
        cls.model_path = BACK_DIR / "model" / "best_model.pth"

    def test_resize_pad_uses_modeling_rounding(self):
        """비정사각형 입력 크기에 모델링과 같은 반올림을 적용합니다."""
        image = np.ones((3, 5), dtype=np.uint8)
        padded = resize_and_pad(image, (8, 8))

        self.assertEqual(padded.shape, (8, 8))
        self.assertEqual(int(np.count_nonzero(padded)), 40)

    def test_preprocessing_builds_one_channel(self):
        """Resize/Pad 전처리가 1채널 범주형 입력을 생성합니다."""
        image = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        array = preprocess_wafer_map(
            image,
            resize_mode="resize_pad",
            target_size=(64, 64),
        )

        self.assertEqual(array.shape, (1, 64, 64))
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(np.isin(array, [0.0, 0.5, 1.0]).all())

    def test_preprocessing_rejects_invalid_maps(self):
        """잘못된 차원·범주·빈 웨이퍼 맵을 거부합니다."""
        invalid_maps = (
            np.ones((2, 3, 1), dtype=np.uint8),
            np.array([[0, 1, 3]], dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
        )
        for image in invalid_maps:
            with (
                self.subTest(shape=image.shape),
                self.assertRaises(ValueError),
            ):
                preprocess_wafer_map(image)

    def test_selected_checkpoint_produces_nine_logits(self):
        """최종 ResidualCNN이 1채널 입력에서 9개 logit을 생성합니다."""
        model, spec = load_model(self.model_path, self.device)
        outputs = model(torch.zeros((2, 1, *TARGET_SIZE)))

        self.assertIsInstance(model, ResidualCNN)
        self.assertEqual(spec.model_name, "residual_cnn")
        self.assertEqual(spec.in_channels, 1)
        self.assertEqual(spec.resize_mode, "resize_pad")
        self.assertEqual(spec.classes, tuple(CLASSES))
        self.assertEqual(tuple(outputs.shape), (2, 9))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 548_361)

    def test_invalid_checkpoint_contracts_are_rejected(self):
        """모델명·전처리·클래스·입력 채널 불일치를 거부합니다."""
        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )
        invalid_checkpoints = []

        wrong_model = copy.deepcopy(checkpoint)
        wrong_model["config"]["model"] = "unknown_model"
        invalid_checkpoints.append(wrong_model)

        wrong_preprocessing = copy.deepcopy(checkpoint)
        wrong_preprocessing["config"]["preprocessing"] = "fixed_resize"
        invalid_checkpoints.append(wrong_preprocessing)

        wrong_classes = copy.deepcopy(checkpoint)
        wrong_classes["classes"] = list(reversed(CLASSES))
        invalid_checkpoints.append(wrong_classes)

        wrong_channels = copy.deepcopy(checkpoint)
        wrong_channels["model_state_dict"]["features.0.0.weight"] = (
            torch.zeros((32, 2, 3, 3))
        )
        invalid_checkpoints.append(wrong_channels)

        with tempfile.TemporaryDirectory() as directory:
            for index, invalid in enumerate(invalid_checkpoints):
                with self.subTest(index=index):
                    path = Path(directory) / f"invalid-{index}.pth"
                    torch.save(invalid, path)
                    with self.assertRaises(ValueError):
                        load_model(path, self.device)

    def test_raw_state_dict_is_rejected(self):
        """메타데이터가 없는 raw state dict를 거부합니다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pth"
            torch.save(ResidualCNN().state_dict(), path)
            with self.assertRaises(ValueError):
                load_model(path, self.device)

    @staticmethod
    def _image_buffer(value):
        """테스트용 grayscale PNG 버퍼를 만듭니다.

        Args:
            value: 이미지 전체에 채울 웨이퍼 범주값입니다.

        Returns:
            PNG 형식의 메모리 버퍼입니다.
        """
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
                    np.ones((20, 30), dtype=np.uint8),
                    np.full((25, 25), 2, dtype=np.uint8),
                ],
                "failureType": ["none", "none"],
                "lotName": ["lotA", "lotA"],
            }
        )
        result = process_labeled_images(
            dataframe,
            self.model_path,
            batch_size=2,
            num_workers=0,
        )

        self.assertEqual(len(result), 8)
        self.assertEqual(len(result[5]), 2)
        self.assertEqual(len(result[6]), 2)
        self.assertEqual(result[7], ["lotA", "lotA"])
        self.assertTrue(
            all(
                0 <= prediction < len(CLASSES)
                for prediction in result[6]
            )
        )


if __name__ == "__main__":
    unittest.main()
