from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from config import settings
from services.llm import ExplanationResult, WaferExplanationService, get_service

router = APIRouter(prefix="/explanation", tags=["explanation"])


class ProcessHistoryItem(BaseModel):
    stepName: str
    tool: str
    recipe: str


class LotSummary(BaseModel):
    lotName: str
    defectCounts: Dict[str, int]
    normalCount: int
    defectiveCount: int
    defectRate: float


class ExplanationResponse(BaseModel):
    lotName: str
    defectRate: float
    explanation: str
    references: List[Dict[str, str]]


# Generates an LLM-backed explanation when the defect rate exceeds the threshold.
@router.post("/get_llm_response", response_model=ExplanationResponse)
async def get_llm_response_api(
    lot: LotSummary,
    service: WaferExplanationService = Depends(get_service),
) -> ExplanationResponse:
    if lot.defectRate < settings.LLM_MIN_DEFECT_THRESHOLD:
        raise HTTPException(status_code=400, detail="Defect rate below explanation threshold.")

    result: ExplanationResult = await service.explain(lot_summary=lot.model_dump())
    return ExplanationResponse(
        lotName=lot.lotName,
        defectRate=lot.defectRate,
        explanation=result.text,
        references=result.references,
    )
