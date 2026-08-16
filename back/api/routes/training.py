import os
import shutil
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import pandas as pd

from core.model.trainer import process
from config.paths import BEST_MODEL_PATH

router = APIRouter()

class AnalysisRequest(BaseModel):
    file_location: str
    f1_score: float

@router.post("/retrain_predict_labeled_images")
async def retrain_predict_labeled_images(data: AnalysisRequest, isPresentation: bool = Query(False)):
    """모델 재학습 및 예측"""
    try:
        dataset = await run_in_threadpool(pd.read_pickle, data.file_location)

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

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))