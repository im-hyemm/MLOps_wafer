import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader

from core.model.architecture import load_model
from core.data.dataset import WaferInferenceDataset
from core.evaluation.metrics import calculate_metrics
from core.evaluation.visualization import draw_cm_heatmap
from utils.image_utils import resize_and_pad
from utils.file_utils import build_dataset_from_zip_stream
from utils.common import get_lot_defect_ranking_with_error_counts
from config.paths import BEST_MODEL_PATH, WEIGHT_USED_MODEL_HEATMAP_PATH
from config.settings import CLASSES, LABEL2ID

def process_one_image(image_bytes_io): 
    """단일 이미지 추론"""
    image = Image.open(image_bytes_io)
    image = np.array(image)

    preprocessed_image = resize_and_pad(image)
    tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).unsqueeze(0).float() / 2.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(BEST_MODEL_PATH, device)

    with torch.no_grad():
        outputs = model(tensor.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    isDefect = (predicted_idx != 8)

    return isDefect.item(), predicted_idx.item(), confidence.item()

def process_multi_images(zip_image):
    """여러 이미지 추론"""
    df_image = build_dataset_from_zip_stream(zip_image)
    df_image['waferMap'] = df_image['waferMap'].apply(resize_and_pad)

    X = df_image['waferMap']
    X_batch = np.stack(X.values)
    X_batch = np.expand_dims(X_batch, axis=1)
    X_tensor = torch.from_numpy(X_batch).float() / 2.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(BEST_MODEL_PATH, device)
    
    with torch.no_grad():
        outputs = model(X_tensor)

    predictions = torch.argmax(outputs, dim=1)
    predictions_np = predictions.cpu().numpy()
    df_image['pred_value'] = predictions_np

    pred_list = predictions_np.tolist()
    lot_defect_ranking = get_lot_defect_ranking_with_error_counts(df_image)

    file_count = len(df_image)
    normal_count = np.sum(predictions_np == 8)
    defective_count = file_count - int(normal_count)
    defect_rate = defective_count / file_count if file_count > 0 else 0.0

    return lot_defect_ranking, file_count, int(normal_count), defective_count, defect_rate, pred_list

def process_multi_images_one_lot(zip_image):
    """단일 lot 기반 여러 이미지 추론"""
    df_image = build_dataset_from_zip_stream(zip_image)
    df_image['waferMap'] = df_image['waferMap'].apply(resize_and_pad)

    X = df_image['waferMap']
    X_batch = np.stack(X.values)
    X_batch = np.expand_dims(X_batch, axis=1)
    X_tensor = torch.from_numpy(X_batch).float() / 2.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(BEST_MODEL_PATH, device)
    
    with torch.no_grad():
        outputs = model(X_tensor)

    predictions = torch.argmax(outputs, dim=1)
    predictions_np = predictions.cpu().numpy()
    df_image['pred_value'] = predictions_np

    pred_list = predictions_np.tolist()
    lot_number = df_image['lotName'].iloc[0] if 'lotName' in df_image.columns else 'Unknown'
    
    defect_type_counts = {i: 0 for i in range(8)}
    for label in pred_list:
        if label in defect_type_counts:
            defect_type_counts[label] += 1
    
    file_count = len(df_image)
    normal_count = np.sum(predictions_np == 8)
    defective_count = file_count - int(normal_count)
    defect_rate = defective_count / file_count if file_count > 0 else 0.0

    return lot_number, defect_type_counts, file_count, int(normal_count), defective_count, round(defect_rate, 2), pred_list

def process_labeled_images(pickle_file, model_location, batch_size=512, num_workers=2):
    """라벨이 있는 데이터 추론 및 평가"""
    if 'waferMap' not in pickle_file or pickle_file['waferMap'].empty:
        print("Warning: 'waferMap' data is empty. Skipping prediction.")
        return None, None, None, None, {}, [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_location, device)
    model.eval()

    all_lots = pickle_file['lotName'].tolist() if 'lotName' in pickle_file else []

    ds = WaferInferenceDataset(pickle_file['waferMap'])
    pin_mem = True if device.type == "cuda" else False
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_mem)

    all_preds = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())

    predictions_np = torch.cat(all_preds).numpy()
    pickle_file['pred_value'] = predictions_np

    test_acc = test_macro_f1 = test_macro_precision = test_macro_recall = None
    metrics_by_cat = {}
    all_labels = []

    if 'failureType' in pickle_file:
        pickle_file['label_id'] = pickle_file['failureType'].map(LABEL2ID)
        y = pickle_file['label_id'].values.astype(int)
        all_labels = y.tolist()

        test_acc, test_macro_f1, test_macro_precision, test_macro_recall, metrics_by_cat = calculate_metrics(y, predictions_np)

        if WEIGHT_USED_MODEL_HEATMAP_PATH and len(np.unique(y)) > 1:
            try:
                _ = draw_cm_heatmap(all_labels, predictions_np, WEIGHT_USED_MODEL_HEATMAP_PATH,
                                    palette='Blues', normalize='true')
            except Exception as e:
                print(f"Confusion Matrix plotting skipped due to error: {e}")
        elif WEIGHT_USED_MODEL_HEATMAP_PATH:
            print("Skipping Confusion Matrix: not enough unique true labels.")
    else:
        print("Warning: 'failureType' not found. Metrics are not computed.")

    return (test_acc, test_macro_f1, test_macro_precision, test_macro_recall,
            metrics_by_cat, all_labels, predictions_np.tolist(), all_lots)