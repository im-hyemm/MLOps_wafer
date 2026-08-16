import os
from fastapi import FastAPI

from api.routes.prediction import router as prediction_router
from api.routes.training import router as training_router
from api.routes.explanation import router as explanation_router
from api.middleware import setup_cors
from config.paths import DATA_DIR, PLOT_DIR

app = FastAPI()

# api key 설정

# 디렉토리 설정
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# CORS 설정
setup_cors(app)

# 라우터 등록
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(explanation_router)

# 실행 명령어: python -m uvicorn main:app --port 8001 --reload
