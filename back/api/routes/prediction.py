import os
import base64
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from core.model.predictor import process_one_image, process_multi_images, process_multi_images_one_lot, process_labeled_images
from utils.image_utils import convert_into_colored_img
from config.paths import BEST_MODEL_PATH
from config.settings import CLASSES
import pandas as pd
from services.db import fetch_lot_process_history

router = APIRouter()

@router.post("/predict_img")
async def predict_img(file: UploadFile = File(...)):
    """단일 이미지 예측"""
    try:
        filename_no_ext = os.path.splitext(os.path.basename(file.filename))[0]
        lot_name, wafer_idx = filename_no_ext.split('_')
        
        image_bytes = await file.read()
        image_bytes_io_colored = BytesIO(image_bytes)
        image_bytes_io_pred = BytesIO(image_bytes)

        colored_img_buf = convert_into_colored_img(image_bytes_io_colored)
        colored_png_b64 = base64.b64encode(colored_img_buf.getvalue()).decode("ascii")
        colored_data_url = f"data:image/png;base64,{colored_png_b64}"
        
        isDefect, label_idx, conf = process_one_image(image_bytes_io_pred)

        lot_process_history = await run_in_threadpool(
            fetch_lot_process_history, lot_name
        )

        return {
            "colored_img": colored_data_url,
            "lotName": lot_name,
            "waferIndex": wafer_idx,
            "isDefect": isDefect,
            "prediction": CLASSES[label_idx],
            "confidence": conf,
            "lotProcessHistory": lot_process_history,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@router.post("/predict_multi_images_one_lot")
async def predict_multi_images_one_lot(file: UploadFile = File(...)):
    """단일 lot 다중 이미지 예측"""
    try:
        zip_bytes = await file.read()
        zip_stream = BytesIO(zip_bytes)

        lot_number, defect_type_counts, file_count, normal_count, defective_count, defect_rate, pred_list = process_multi_images_one_lot(zip_stream)
        
        lot_process_history = await run_in_threadpool(
            fetch_lot_process_history, lot_number
        )

        return {
            "lotNumber": lot_number,
            "defectTypeCounts": defect_type_counts,
            "fileCount": file_count,
            "normal": normal_count,
            "defective": defective_count,
            "defectRate": defect_rate,
            "predictions": pred_list,
            "lotProcessHistory": lot_process_history,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@router.post("/predict_multi_images_multi_lots")
async def predict_multi_images_multi_lots(file: UploadFile = File(...)):
    """다중 lot 다중 이미지 예측"""
    try:
        zip_bytes = await file.read()
        zip_stream = BytesIO(zip_bytes)

        lot_defect_ranking, file_count, normal_count, defective_count, defect_rate, pred_list = process_multi_images(zip_stream)

        lot_summaries = []
        lot_process_history_map = {}
        for lot_name, error_detail in lot_defect_ranking.items():
            lot_history = await run_in_threadpool(
                fetch_lot_process_history, lot_name
            )
            lot_process_history_map[lot_name] = lot_history

            total_count = sum(error_detail.values())
            normal_per_lot = error_detail.get(8, 0)
            defective_per_lot = total_count - normal_per_lot
            defect_rate_per_lot = defective_per_lot / total_count if total_count else 0.0

            defect_counts = {}
            for label_idx, count in error_detail.items():
                if 0 <= label_idx < len(CLASSES):
                    label_name = CLASSES[label_idx]
                else:
                    label_name = str(label_idx)
                defect_counts[label_name] = count

            process_history = []
            for step_name, details in (lot_history or {}).items():
                details = details or {}
                process_history.append(
                    {
                        "stepName": step_name,
                        "tool": details.get("tool"),
                        "recipe": details.get("recipe"),
                    }
                )

            lot_summaries.append(
                {
                    "lotName": lot_name,
                    "defectCounts": defect_counts,
                    "normalCount": normal_per_lot,
                    "defectiveCount": defective_per_lot,
                    "defectRate": defect_rate_per_lot,
                    "processHistory": process_history,
                }
            )

        return {
            "filename": file.filename,
            "lot_defect_ranking": lot_defect_ranking,
            "fileCount": file_count,
            "normal": normal_count,
            "defective": defective_count,
            "defectRate": defect_rate,
            "predictions": pred_list,
            "lots": lot_summaries,
            "lotProcessHistory": lot_process_history_map,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@router.post("/upload_predict_labeled_images")
async def upload_predict_labeled_images(file: UploadFile = File(...)):
    """라벨된 이미지 업로드 및 예측"""
    try:
        from datetime import datetime
        from config.settings import TIMEZONE
        from config.paths import DATA_DIR
        
        current_time = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = os.path.join(DATA_DIR, new_filename)

        with open(file_location, "wb") as f:
            f.write(await file.read())

        dataset = await run_in_threadpool(pd.read_pickle, file_location)

        nrow = dataset.shape[0]
        nlot = len(dataset['lotName'].unique())
        labeled_ratio = round(dataset['failureType'].notna().sum() / nrow * 100, 2)
        normal_num = dataset[dataset['failureType'] == "none"].shape[0]
        defect_num = nrow - normal_num
        defect_ratio = round(defect_num / nrow * 100, 2)
        num_by_cat = dataset['failureType'].value_counts().to_dict()

        (test_acc_used, test_macro_f1_used, test_macro_precision_used, test_macro_recall_used, 
         metrics_by_cat_used, all_labels_used, all_preds_used, all_lots_used) = await run_in_threadpool(
             process_labeled_images, dataset, model_location=BEST_MODEL_PATH
             )
        
        model_retrain = test_macro_f1_used < 0.7
        
        response = {
            "file_location": file_location,
            "model_retrain": model_retrain,
            "meta": {
                "nrow": nrow,
                "nlot": nlot,
                "labeled_ratio": labeled_ratio,
                "defect_ratio": defect_ratio,
                "num_by_cat": num_by_cat
            },
            "used": {
                "overal_acc": test_acc_used,
                "macro_f1": test_macro_f1_used,
                "macro_precision": test_macro_precision_used,
                "macro_recall": test_macro_recall_used,
                "labels": all_labels_used,
                "preds": all_preds_used,
                "lots": all_lots_used,
                "metrics_by_cat": metrics_by_cat_used
            },
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))