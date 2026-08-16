"""LangChain-powered helpers for wafer defect explanations (refactored for given settings)."""
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from config import settings
from config.paths import PAPER_DIR, INDEX_DIR, TRACKING_FILE_PATH
from scripts.build_paper_index import rebuild_index


@dataclass
class ExplanationResult:
    text: str
    references: List[Dict[str, Any]]


def _safe_float_str(x: Any) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _format_docs_for_context(docs: Sequence[Any]) -> str:
    """문헌 컨텍스트를 LLM에 투입하기 좋게 정리."""
    chunks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        title = meta.get("title") or meta.get("source") or meta.get("file_name") or "Untitled"
        authors = meta.get("authors") or meta.get("author") or ""
        year = meta.get("year") or ""
        pages = meta.get("page") or meta.get("pages") or ""
        head = f"[{i}] {title} ({authors}, {year}) p.{pages}".strip()
        chunks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(chunks)


def _extract_references(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    """UI/후속 로깅용 참고문헌 메타데이터."""
    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append({k: (str(v) if v is not None else "") for k, v in meta.items()})
    return out


class WaferExplanationService:
    """Generates LLM-based explanations for wafer defect rates with structured Korean output."""

    def __init__(
        self,
        paper_dir: Path,
        index_dir: Path,
        tracking_file: Path,
        model_name: str,
        embedding_model: str,
        top_k: int,
        refresh_interval: float,
        temperature: float,
    ) -> None:
        self.paper_dir = paper_dir
        self.index_dir = index_dir
        self.tracking_file = tracking_file
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.refresh_interval = refresh_interval
        self._retriever: Optional[Any] = None
        self._refresh_lock = asyncio.Lock()
        self._last_refresh: Optional[float] = None
        self.temperature = temperature

        # 설정에 맞춰 고정(temperature 등 추가 필드 없음)
        self._llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=temperature,
        )

        # 한국어 구조화 출력 강제 프롬프트(섹션 고정 + 인용 표기)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "당신은 반도체 공정 엔지니어입니다. "
                        "당신의 임무는 웨이퍼맵 불량 유형 분포와 관련 문헌(학술/산업 보고서) 근거를 참고하여 "
                        "현장에서 즉시 활용 가능한 간단한 알림(notice) 형태의 메시지를 작성하는 것입니다. "
                        "출력은 한국어로 하십시오. "
                        "보고서처럼 길게 작성하지 말고, 작업자가 한눈에 확인할 수 있도록 간결하고 실무적인 표현만 사용하십시오. "
                        "참고 문헌 번호나 인용은 포함하지 마십시오."
                        "출력은 아래 형식 및 규칙을 반드시 준수합니다. \n\n"

                        "### 반드시 지켜야 할 규칙\n"
                        "1) 가능한 불량 유형 목록은 top_defects에 한정됩니다. top_defects에 없는 유형은 절대 쓰지 마십시오.\n"
                        "2) '🔔 Possible Cause:' 아래에는 top_defects 중 개수가 많은 순으로 나열하십시오.\n"
                        "3) '✅ Checkpoints:'는 상위 불량 유형들을 우선 고려해 3~5개 불릿으로 작성하십시오(동사로 시작·즉시 수행 가능 항목).\n"
                        "4) '📌 Action Priority:'는 defect_rate(숫자)로만 결정하고, 아래 규칙표의 문구를 정확히 그대로 사용하십시오.\n"
                        "   - defect_rate ≥ 0.20  → '즉시 점검 필요(매우 심각)'\n"
                        "   - 0.10 ≤ defect_rate < 0.20 → '우선 점검 필요(높음)'\n"
                        "   - 0.05 ≤ defect_rate < 0.10 → '단기 점검 권장(중간)'\n"
                        "   - min_threshold ≤ defect_rate < 0.05 → '모니터링 강화(낮음)'\n"
                        "   - defect_rate < min_threshold → 전체 카드를 출력하지 말고 "
                        "'📌 Action Priority: 모니터링 권장 (임계치 미만으로 과도 해석 지양)' 한 줄만 출력하십시오.\n"
                        "5) 형식 외 텍스트·헤더·표·코드블록을 추가하지 마십시오.\n"
                    ),
                ),
                (
                    "human",
                    (
                        "검색된 문헌 요약:\n\n"
                        "{context}\n\n"
                        "분석 대상 로트 개요 및 메모:\n\n"
                        "Lot name: {lot_name}\n"
                        "Defect rate: {defect_rate}\n"
                        "LLM 최소 해석 임계치(참고): {min_threshold}\n"
                        "주요 불량 유형(top_defects): {top_defects}\n\n"
                        "Lot summary (JSON):\n{formatted_summary}\n\n"
                        "Process notes:\n{notes_text}\n\n"
                        "요청:\n"
                        "1) '🔔 Possible Cause:' 아래에 불량 유형별 원인을 줄글로 나열하십시오.\n"
                        "   - 형식 예: 'Edge-Loc 불량 - 장비 정렬 불량, 웨이퍼 이송 문제'\n"
                        "   - 불량 개수가 0개인 것은 포함하지 마시오.\n"
                        "   - 최대 3개 유형까지만 표시합니다.\n"
                        "2) '✅ Checkpoints:' 아래에는 불량 개수가 많은 유형 순서대로 가장 중요한 점검 항목을 3~5개 불릿으로 정리하십시오.\n"
                        "   - 항목은 동사로 시작하고 실무적으로 바로 수행 가능한 형태로 작성하십시오.\n"
                        "3) 마지막에 '📌 Action Priority:'를 로트 전체 불량률(defect_rate)과 규칙 표에 따라 정확히 산출하여 한 줄로만 제시하십시오.\n"
                        "4) defect_rate가 min_threshold보다 낮으면, 전체 카드를 출력하지 말고 다음 한 줄만 출력하십시오:\n"
                        "'📌 Action Priority: 모니터링 권장 (임계치 미만으로 과도 해석 지양)'\n"
                    ),
                ),
                (
                    "system",
                    (
                        "### 출력 형식(정확히 준수)\n"
                        "🔔 Possible Cause:\n"
                        "- {{유형A}} 불량 - {{원인키워드A-1}}, {{원인키워드A-2}}\n"
                        "- {{유형B}} 불량 - {{원인키워드B-1}}, {{원인키워드B-2}}\n"
                        "\n"
                        "✅ Checkpoints:\n"
                        "- {{점검 항목 1}}\n"
                        "- {{점검 항목 2}}\n"
                        "- {{점검 항목 3}}\n"
                        "- {{점검 항목 4}}\n"
                        "- {{점검 항목 5}}\n"
                        "\n"
                        "📌 Action Priority: {{규칙표에 따른 정확한 문구}}"
                    ),
                ),
            ]
        )

        self._parser = StrOutputParser()

    async def _load_retriever(self) -> None:
        index_path = self.index_dir / "index.faiss"
        if not index_path.exists():
            raise RuntimeError(
                f"Vector store not found at {index_path}. Run scripts/build_paper_index.py to create it."
            )
        embeddings = OpenAIEmbeddings(model=settings.LLM_EMBEDDING_MODEL)
        vector_store = await asyncio.to_thread(
            FAISS.load_local,
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self._retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def _format_lot_summary(self, lot_summary: Dict[str, Any]) -> str:
        try:
            return json.dumps(lot_summary, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(lot_summary)

    def _build_question_inputs(self, lot_summary: Dict[str, Any], notes: Optional[str]) -> Dict[str, str]:
        notes_text = (notes or "").strip() or "None provided."
        lot_name = lot_summary.get("lotName") or "unknown lot"
        defect_rate = _safe_float_str(lot_summary.get("defectRate"))
        formatted_summary = self._format_lot_summary(lot_summary)
        min_threshold = _safe_float_str(getattr(settings, "LLM_MIN_DEFECT_THRESHOLD", 0.0))
        
         # 🔑 검색용 쿼리 생성
        defects: Dict[str, int] = lot_summary.get("defectCounts", {}) or {}
        top_defects = [t for t, c in sorted(defects.items(), key=lambda x: x[1], reverse=True) if c > 0 and str(t).lower() != "none"][:3]

        query_parts = []
        if top_defects:
            query_parts.append("Defect types: " + ", ".join(top_defects))
        query_parts.append(f"Defect rate: {defect_rate}")
        if notes_text and notes_text != "None provided.":
            query_parts.append("Notes: " + notes_text)

        # 최종 검색 질문
        query = "; ".join(query_parts) + "; semiconductor wafer root cause and inspection checkpoints"

        return {
            "lot_name": lot_name,
            "defect_rate": defect_rate,
            "formatted_summary": formatted_summary,
            "notes_text": notes_text,
            "min_threshold": min_threshold,
            "top_defects": top_defects,
            "query": query,   # retriever에 넘길 핵심 문자열
        }

    async def refresh_vector_store_if_needed(self, force: bool = False) -> bool:
        async with self._refresh_lock:
            now = time.time()
            needs_refresh = (
                force
                or self._retriever is None
                or self._last_refresh is None
                or (now - self._last_refresh) >= self.refresh_interval
            )
            if not needs_refresh:
                return False

            updated = await asyncio.to_thread(
                rebuild_index,
                self.paper_dir,
                self.index_dir,
                self.tracking_file,
            )
            if updated or self._retriever is None:
                await self._load_retriever()
            self._last_refresh = time.time()
            return updated

    async def explain(self, lot_summary: Dict[str, Any], notes: Optional[str] = None) -> ExplanationResult:
        # 1) 인덱스 최신화/로딩
        await self.refresh_vector_store_if_needed()
        if self._retriever is None:
            raise RuntimeError("Vector store is not available. Ensure papers have been indexed.")

        # 2) 문헌 검색
        question_inputs = self._build_question_inputs(lot_summary, notes)
        query = question_inputs["query"] 
        docs = await self._retriever.ainvoke(query)

        # 3) 컨텍스트 생성
        context = _format_docs_for_context(docs)


        # 4) LCEL 파이프라인 실행: prompt -> llm -> text
        chain = self._prompt | self._llm | self._parser
        text = await chain.ainvoke({**question_inputs, "context": context})


        # 5) 참고문헌 메타데이터 구성
        references = _extract_references(docs)

        return ExplanationResult(text=text, references=references)


_service: Optional[WaferExplanationService] = None


def get_service() -> WaferExplanationService:
    global _service
    if _service is None:
        _service = WaferExplanationService(
            paper_dir=Path(PAPER_DIR).resolve(),
            index_dir=Path(INDEX_DIR).resolve(),
            tracking_file=Path(TRACKING_FILE_PATH).resolve(),
            model_name=settings.LLM_EXPLANATION_MODEL,
            embedding_model=settings.LLM_EMBEDDING_MODEL,
            top_k=settings.LLM_EXPLANATION_TOP_K,
            refresh_interval=settings.LLM_INDEX_REFRESH_INTERVAL_SECONDS,
            temperature=settings.LLM_EXPLANATION_TEMPERATURE
        )
    return _service


async def explain_defect_rate(lot_summary: Dict[str, Any], notes: Optional[str] = None) -> ExplanationResult:
    service = get_service()
    return await service.explain(lot_summary=lot_summary, notes=notes)
