from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.routes.explanation import router
from services.llm import ExplanationResult, get_service

app = FastAPI()
app.include_router(router)
client = TestClient(app)


class DummyService:
    async def explain(self, lot_summary, notes=None):
        return ExplanationResult(text="dummy-explanation", references=[{"source": "test"}])


@pytest.fixture(autouse=True)
def override_dependency():
    previous = app.dependency_overrides.get(get_service)
    app.dependency_overrides[get_service] = lambda: DummyService()
    yield
    if previous is None:
        app.dependency_overrides.pop(get_service, None)
    else:
        app.dependency_overrides[get_service] = previous


def test_explanation_requires_threshold():
    payload = {
        "lotName": "LOT-001",
        "defectCounts": {"Center": 3},
        "normalCount": 7,
        "defectiveCount": 3,
        "defectRate": 0.05,
        "processHistory": [],
    }
    response = client.post("/explanation/get_llm_response", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Defect rate below explanation threshold."


def test_explanation_success():
    payload = {
        "lotName": "LOT-002",
        "defectCounts": {"Center": 12},
        "normalCount": 30,
        "defectiveCount": 12,
        "defectRate": 0.20,
        "processHistory": [
            {"stepName": "STEP", "tool": "TOOL", "recipe": "RECIPE"},
        ],
    }
    response = client.post("/explanation/get_llm_response", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] == "dummy-explanation"
    assert body["references"] == [{"source": "test"}]
