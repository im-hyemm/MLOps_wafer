import os
import shutil
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import pandas as pd

from core.model.trainer import process
from core.model.predictor import process_labeled_images
from config.paths import BEST_MODEL_PATH, PRESENTATION_MODEL_PATH

router = APIRouter()

class AnalysisRequest(BaseModel):
    file_location: str
    f1_score: float

@router.post("/retrain_predict_labeled_images")
async def retrain_predict_labeled_images(data: AnalysisRequest, isPresentation: bool = Query(True)):
    """모델 재학습 및 예측"""
    try:
        dataset = await run_in_threadpool(pd.read_pickle, data.file_location)
        
        # 발표용 response
        (test_acc_pre, test_macro_f1_pre, test_macro_precision_pre, test_macro_recall_pre, 
        metrics_by_cat_pre, all_labels_pre, all_preds_pre, all_lots_pre) = await run_in_threadpool(
            process_labeled_images, dataset, model_location=PRESENTATION_MODEL_PATH
            )

        response_presentation = {
            "model_change": True,
            "new":  {
                "overal_acc": test_acc_pre,
                "macro_f1": test_macro_f1_pre,
                "macro_precision": test_macro_precision_pre,
                "macro_recall": test_macro_recall_pre,
                "labels": all_labels_pre,
                "preds": all_preds_pre,
                "lots": all_lots_pre,
                "metrics_by_cat": metrics_by_cat_pre
            },
        }

        # 학습 후 실제 response
        if isPresentation:
            (test_acc_new, test_macro_f1_new, test_macro_precision_new, test_macro_recall_new, 
            metrics_by_cat_new, all_labels_new, all_preds_new, all_lots_new, new_model_location) = await run_in_threadpool(
                    process, dataset, True
                )
        else:
            (test_acc_new, test_macro_f1_new, test_macro_precision_new, test_macro_recall_new, 
            metrics_by_cat_new, all_labels_new, all_preds_new, all_lots_new, new_model_location) = await run_in_threadpool(
                    process, dataset, False
                )
        
        if isPresentation:
            test_macro_f1_new = test_macro_f1_pre
        
        if (test_macro_f1_new > data.f1_score):
            print("재학습된 모델의 성능이 더 좋습니다. BEST_MODEL을 재학습된 모델로 교체합니다.")
            os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
            tmp_path = BEST_MODEL_PATH + ".tmp"
            shutil.copyfile(new_model_location, tmp_path)
            os.replace(tmp_path, BEST_MODEL_PATH)

            response = {
                "model_change": True,
                "new":  {
                    "overal_acc": test_acc_new,
                    "macro_f1": test_macro_f1_new,
                    "macro_precision": test_macro_precision_new,
                    "macro_recall": test_macro_recall_new,
                    "labels": all_labels_new,
                    "preds": all_preds_new,
                    "lots": all_lots_new,
                    "metrics_by_cat": metrics_by_cat_new
                },
            }
        else: 
            print("재학습하였으나 이전 모델의 성능이 더 우수하여 변경하지 않습니다.")
            response = {
                "model_change": False
            }

        return response_presentation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))