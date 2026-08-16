import os
from pathlib import Path

# 기본 경로 설정
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = os.getenv("BASE_DIR", str(DEFAULT_BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data/")
MODEL_DIR = os.path.join(BASE_DIR, "model/")
PLOT_DIR = os.path.join(BASE_DIR, "plots/")
PAPER_DIR = os.path.join(BASE_DIR, "paper/")
INDEX_DIR = os.path.join(BASE_DIR, "vector_store/")

# 파일 경로
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
PRESENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "presentation_model.pth")
WEIGHT_USED_MODEL_HEATMAP_PATH = os.path.join(PLOT_DIR, "weight_used_model_heatmap.html")
NEW_MODEL_HEATMAP_PATH = os.path.join(PLOT_DIR, "new_model_heatmap.html")
TRACKING_FILE_PATH = os.path.join(INDEX_DIR, "processed_files.txt")
