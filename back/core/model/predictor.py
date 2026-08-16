import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config.paths import BEST_MODEL_PATH, WEIGHT_USED_MODEL_HEATMAP_PATH
from config.settings import LABEL2ID
from core.data.dataset import WaferInferenceDataset
from core.evaluation.metrics import calculate_metrics
from core.evaluation.visualization import draw_cm_heatmap
from core.model.architecture import load_model
from utils.common import get_lot_defect_ranking_with_error_counts
from utils.file_utils import build_dataset_from_zip_stream
from utils.image_utils import preprocess_wafer_map


def _load_runtime_model(model_path):
    """실행 장치에 모델을 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = load_model(model_path, device)
    return model, spec, device


def _build_input_batch(images, spec):
    """웨이퍼 맵 목록을 모델 입력 배치로 변환합니다."""
    arrays = [
        preprocess_wafer_map(
            image,
            resize_mode=spec.resize_mode,
            target_size=spec.target_size,
        )
        for image in images
    ]
    return torch.from_numpy(np.stack(arrays)).float()


def process_one_image(image_bytes_io):
    """단일 이미지의 불량 여부와 클래스를 추론합니다.

    Args:
        image_bytes_io: 업로드 이미지 바이트 스트림입니다.

    Returns:
        불량 여부, 클래스 ID, confidence의 튜플입니다.
    """
    model, spec, device = _load_runtime_model(BEST_MODEL_PATH)
    image = np.array(Image.open(image_bytes_io))
    tensor = _build_input_batch([image], spec).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    is_defect = predicted_idx != 8
    return is_defect.item(), predicted_idx.item(), confidence.item()


def process_multi_images(zip_image):
    """여러 Lot의 이미지 ZIP을 추론합니다.

    Args:
        zip_image: 업로드 ZIP 바이트 스트림입니다.

    Returns:
        Lot 순위와 전체 예측 통계의 튜플입니다.
    """
    dataframe = build_dataset_from_zip_stream(zip_image)
    model, spec, device = _load_runtime_model(BEST_MODEL_PATH)
    input_tensor = _build_input_batch(
        dataframe["waferMap"].tolist(),
        spec,
    ).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    predictions = torch.argmax(outputs, dim=1)
    predictions_np = predictions.cpu().numpy()
    dataframe["pred_value"] = predictions_np

    prediction_list = predictions_np.tolist()
    lot_defect_ranking = get_lot_defect_ranking_with_error_counts(dataframe)
    file_count = len(dataframe)
    normal_count = int(np.sum(predictions_np == 8))
    defective_count = file_count - normal_count
    defect_rate = defective_count / file_count if file_count > 0 else 0.0

    return (
        lot_defect_ranking,
        file_count,
        normal_count,
        defective_count,
        defect_rate,
        prediction_list,
    )


def process_multi_images_one_lot(zip_image):
    """단일 Lot의 이미지 ZIP을 추론합니다.

    Args:
        zip_image: 업로드 ZIP 바이트 스트림입니다.

    Returns:
        Lot 번호와 불량 유형별 통계의 튜플입니다.
    """
    dataframe = build_dataset_from_zip_stream(zip_image)
    model, spec, device = _load_runtime_model(BEST_MODEL_PATH)
    input_tensor = _build_input_batch(
        dataframe["waferMap"].tolist(),
        spec,
    ).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    predictions = torch.argmax(outputs, dim=1)
    predictions_np = predictions.cpu().numpy()
    dataframe["pred_value"] = predictions_np

    prediction_list = predictions_np.tolist()
    lot_number = (
        dataframe["lotName"].iloc[0]
        if "lotName" in dataframe
        else "Unknown"
    )
    defect_type_counts = {class_id: 0 for class_id in range(8)}
    for label in prediction_list:
        if label in defect_type_counts:
            defect_type_counts[label] += 1

    file_count = len(dataframe)
    normal_count = int(np.sum(predictions_np == 8))
    defective_count = file_count - normal_count
    defect_rate = defective_count / file_count if file_count > 0 else 0.0

    return (
        lot_number,
        defect_type_counts,
        file_count,
        normal_count,
        defective_count,
        round(defect_rate, 2),
        prediction_list,
    )


def process_labeled_images(
    pickle_file,
    model_location,
    batch_size=512,
    num_workers=2,
):
    """라벨 데이터의 추론 결과와 성능을 계산합니다.

    Args:
        pickle_file: 웨이퍼 맵과 선택적 라벨이 포함된 DataFrame입니다.
        model_location: 평가할 체크포인트 경로입니다.
        batch_size: 추론 배치 크기입니다.
        num_workers: DataLoader 워커 수입니다.

    Returns:
        전체 성능과 클래스별 성능, 라벨, 예측, Lot 목록입니다.
    """
    if "waferMap" not in pickle_file or pickle_file["waferMap"].empty:
        return None, None, None, None, {}, [], [], []

    model, spec, device = _load_runtime_model(model_location)
    all_lots = (
        pickle_file["lotName"].tolist()
        if "lotName" in pickle_file
        else []
    )

    dataset = WaferInferenceDataset(
        pickle_file["waferMap"],
        resize_mode=spec.resize_mode,
        target_size=spec.target_size,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            predictions = torch.argmax(model(inputs), dim=1)
            all_predictions.append(predictions.cpu())

    predictions_np = torch.cat(all_predictions).numpy()
    pickle_file["pred_value"] = predictions_np
    metrics = (None, None, None, None)
    metrics_by_category = {}
    all_labels = []

    if "failureType" in pickle_file:
        pickle_file["label_id"] = pickle_file["failureType"].map(LABEL2ID)
        if pickle_file["label_id"].isna().any():
            raise ValueError("지원하지 않는 failureType이 포함되어 있습니다.")
        labels = pickle_file["label_id"].to_numpy(dtype=int)
        all_labels = labels.tolist()
        metric_result = calculate_metrics(
            labels,
            predictions_np,
        )
        metrics = metric_result[:4]
        metrics_by_category = metric_result[4]

        if WEIGHT_USED_MODEL_HEATMAP_PATH and len(np.unique(labels)) > 1:
            try:
                draw_cm_heatmap(
                    all_labels,
                    predictions_np,
                    WEIGHT_USED_MODEL_HEATMAP_PATH,
                    palette="Blues",
                    normalize="true",
                )
            except Exception as error:
                print(f"Confusion Matrix 생성을 건너뜁니다: {error}")

    return (
        *metrics,
        metrics_by_category,
        all_labels,
        predictions_np.tolist(),
        all_lots,
    )
