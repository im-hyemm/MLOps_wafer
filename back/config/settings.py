import pytz

# 기본 설정
SEED = 42
TIMEZONE = pytz.timezone("Asia/Seoul")

# 모델 설정
NUM_CLASSES = 9
TARGET_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30
PRESENTATION_EPOCHS = 2
PRESENTATION_BATCH_SIZE = 10
MODEL_NAME = "residual_cnn"
MODEL_PREPROCESSING = "resize_pad"
RETRAIN_F1_THRESHOLD = 0.7

# 클래스 정의
CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
]
ID2LABEL = {i: label for i, label in enumerate(CLASSES)}
LABEL2ID = {label: i for i, label in enumerate(CLASSES)}

# 학습 설정
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3

# vector store 설정 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# LLM 설정
LLM_EXPLANATION_MODEL: str = "gpt-4o-mini"
LLM_EXPLANATION_TOP_K: int = 5
LLM_EXPLANATION_TEMPERATURE: float = 0.0
LLM_MIN_DEFECT_THRESHOLD: float = 0.10
LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"
LLM_INDEX_REFRESH_INTERVAL_SECONDS: float = 900.0


