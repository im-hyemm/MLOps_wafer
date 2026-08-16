import os
import shutil

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sklearn.metrics import f1_score

from config.paths import BEST_MODEL_PATH
from config.settings import NUM_CLASSES
from core.model.predictor import process_labeled_images
from core.model.trainer import process, split_dataset

router = APIRouter()


class AnalysisRequest(BaseModel):
    """재학습 요청 본문입니다."""

    file_location: str
    f1_score: float


def calculate_raw_macro_f1(labels, predictions):
    """9개 고정 클래스의 반올림 전 Macro F1을 계산합니다.

    Args:
        labels: 정답 클래스 ID 목록입니다.
        predictions: 예측 클래스 ID 목록입니다.

    Returns:
        반올림하지 않은 Macro F1입니다.
    """
    return f1_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )


def should_promote_model(incumbent_f1, candidate_f1):
    """후보 모델의 승격 여부를 판단합니다.

    Args:
        incumbent_f1: 현재 배포 모델의 원시 Macro F1입니다.
        candidate_f1: 재학습 후보 모델의 원시 Macro F1입니다.

    Returns:
        후보 점수가 엄격하게 더 높은지 여부입니다.
    """
    return candidate_f1 > incumbent_f1


def promote_checkpoint(candidate_path, target_path):
    """후보 체크포인트를 대상 경로에 원자적으로 반영합니다.

    Args:
        candidate_path: 승격할 후보 체크포인트 경로입니다.
        target_path: 서비스가 사용하는 체크포인트 경로입니다.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temporary_path = f"{target_path}.tmp"
    try:
        shutil.copyfile(candidate_path, temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


@router.post("/retrain_predict_labeled_images")
async def retrain_predict_labeled_images(
    data: AnalysisRequest,
    is_presentation: bool = Query(False, alias="isPresentation"),
):
    """후보 모델을 재학습하고 동일 테스트셋에서 승격을 판단합니다."""
    try:
        dataset = await run_in_threadpool(pd.read_pickle, data.file_location)
        data_splits = await run_in_threadpool(split_dataset, dataset)
        training_result = await run_in_threadpool(
            process,
            dataset,
            is_presentation,
            data_splits,
        )
        (
            test_accuracy_new,
            test_macro_f1_new,
            test_macro_precision_new,
            test_macro_recall_new,
            metrics_by_category_new,
            all_labels_new,
            all_predictions_new,
            all_lots_new,
            new_model_location,
        ) = training_result

        incumbent_result = await run_in_threadpool(
            process_labeled_images,
            data_splits.test.copy(),
            BEST_MODEL_PATH,
        )
        all_labels_used = incumbent_result[5]
        all_predictions_used = incumbent_result[6]
        if all_labels_used != all_labels_new:
            raise ValueError("현재 모델과 후보 모델의 테스트 라벨이 다릅니다.")
        incumbent_f1 = calculate_raw_macro_f1(
            all_labels_used,
            all_predictions_used,
        )
        candidate_f1 = calculate_raw_macro_f1(
            all_labels_new,
            all_predictions_new,
        )

        if should_promote_model(incumbent_f1, candidate_f1):
            await run_in_threadpool(
                promote_checkpoint,
                new_model_location,
                BEST_MODEL_PATH,
            )
            return {
                "model_change": True,
                "new": {
                    "overal_acc": test_accuracy_new,
                    "macro_f1": test_macro_f1_new,
                    "macro_precision": test_macro_precision_new,
                    "macro_recall": test_macro_recall_new,
                    "labels": all_labels_new,
                    "preds": all_predictions_new,
                    "lots": all_lots_new,
                    "metrics_by_cat": metrics_by_category_new,
                },
            }

        return {"model_change": False}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
