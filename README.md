# WaferGuard

> CNN 기반 웨이퍼 결함 분류와 논문 기반 원인 분석을 결합한 MLOps 서비스

WaferGuard는 반도체 웨이퍼 맵을 분석해 결함 패턴을 분류하고, Lot별 공정 이력과 결함 통계를 제공합니다. 재학습한 모델과 기존 모델의 성능을 비교해 더 나은 모델을 반영하며, 결함률이 높은 Lot에는 관련 논문을 검색해 LLM 기반 원인 설명과 점검 항목을 제공합니다.

## 주요 기능

- 단일 웨이퍼 이미지의 결함 유형 및 신뢰도 예측
- 단일 Lot 및 다중 Lot 단위의 결함률·유형별 통계 분석
- PostgreSQL에 저장된 Lot별 공정 장비 및 레시피 이력 조회
- 라벨 데이터 업로드를 통한 기존 모델 성능 평가
- Macro F1 임계치 기반 수동 재학습 및 동일 테스트셋 모델 승격
- FAISS 문헌 검색과 OpenAI API를 활용한 결함 원인 및 점검 항목 설명

분류 대상은 `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-full`, `none`의 9개 클래스입니다.

## 배포 모델

현재 배포 모델은 웨이퍼 맵과 유효 영역 mask를 함께 사용하는 2채널 CNN입니다. 원본 웨이퍼 맵의 가로세로 비율을 유지해 `64×64`로 리사이즈·패딩하고, 패딩이 아닌 영역을 표시하는 mask를 두 번째 채널로 결합해 `(2, 64, 64)` 입력을 생성합니다.

### 모델 구조

| 단계 | 레이어 | 출력 형태 |
| --- | --- | --- |
| 입력 | 웨이퍼 맵 채널 + 유효 영역 mask 채널 | `2×64×64` |
| 합성곱 블록 1 | `Conv2d(2→32, 3×3)` → BatchNorm → ReLU → MaxPool | `32×32×32` |
| 합성곱 블록 2 | `Conv2d(32→64, 3×3)` → BatchNorm → ReLU → MaxPool | `64×16×16` |
| 합성곱 블록 3 | `Conv2d(64→128, 3×3)` → BatchNorm → ReLU → MaxPool | `128×8×8` |
| 합성곱 블록 4 | `Conv2d(128→256, 3×3)` → BatchNorm → ReLU → AdaptiveAvgPool | `256×1×1` |
| 분류기 | Flatten → Dropout(`0.5`) → Linear(`256→9`) | 클래스 logit 9개 |

학습 시 클래스 평균 크기 기준 오버샘플링과 좌우·상하 반전, ±15도 회전 증강을 적용합니다. 손실 함수는 Cross Entropy, optimizer는 Adam을 사용하며 validation Macro F1을 기준으로 가장 좋은 epoch의 가중치를 선택합니다.

### 모델 선택 근거

현재 모델 구조와 전처리 방식은 [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb)의 비교 실험을 바탕으로 선택했습니다. 해당 노트북에서 고정 크기 resize, 비율 유지 resize·padding, 데이터 증강, 클래스 오버샘플링, 유효 영역 mask 채널의 조합을 비교했으며, 최종적으로 비율 유지 resize·padding과 mask 채널, 오버샘플링, 데이터 증강을 함께 사용하는 구성을 채택했습니다.

- `back/model/best_model.pth`: 서비스에서 사용하는 2채널 모델
- `back/model/best_model_legacy.pth`: 교체 전 1채널 모델의 롤백용 가중치
- `back/model/presentation_model.pth`: 기존 발표용 1채널 모델

모델 로더는 metadata가 포함된 2채널 checkpoint bundle과 기존 raw `state_dict` 형식을 모두 지원합니다. 체크포인트에 따라 2채널 모델에는 반올림 기반 resize와 mask를 적용하고, 기존 1채널 모델에는 종전의 정수 절삭 resize를 적용합니다.

## 시스템 아키텍처

```mermaid
flowchart LR
    U["사용자"] --> F["Vue 3 대시보드"]
    F --> A["FastAPI"]
    A --> M["PyTorch CNN<br/>학습 및 추론"]
    A --> D["PostgreSQL<br/>Lot 공정 이력"]
    A --> R["LangChain + FAISS<br/>문헌 검색"]
    R --> L["OpenAI API<br/>결함 원인 설명"]
```

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| Frontend | Vue 3, Vite, Axios, Chart.js, Bootstrap |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | PyTorch, torchvision, scikit-learn, OpenCV |
| Database | PostgreSQL, psycopg2 |
| LLM/RAG | OpenAI API, LangChain, FAISS, PyMuPDF |

## 프로젝트 구조

```text
MLOps_wafer/
├── back/
│   ├── api/
│   │   └── routes/               # 예측, 재학습, LLM 설명 API
│   ├── config/                   # 환경, 모델, 경로 및 DB 설정
│   ├── core/
│   │   ├── data/                 # 학습·추론 PyTorch Dataset
│   │   ├── evaluation/           # 평가지표와 confusion matrix
│   │   └── model/                # CNN 구조, 추론, 재학습 파이프라인
│   ├── model/
│   │   ├── best_model.pth        # 현재 배포 2채널 모델
│   │   ├── best_model_legacy.pth # 롤백용 기존 1채널 모델
│   │   └── presentation_model.pth
│   ├── paper/                    # RAG 검색 대상 논문
│   ├── scripts/                  # 문헌 인덱스 생성 스크립트
│   ├── services/                 # PostgreSQL 및 LLM 서비스
│   ├── tests/                    # API·모델·재학습 테스트
│   ├── utils/                    # 이미지·파일 공통 기능
│   ├── .env.example
│   ├── create_db.py
│   └── main.py                   # FastAPI 진입점
├── front/
│   ├── src/
│   │   ├── components/           # 공통 UI 컴포넌트
│   │   ├── pages/                # 단일·Lot·pickle 분석 화면
│   │   └── router/               # Vue Router 설정
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   ├── EDA.ipynb                 # 웨이퍼 데이터 탐색
│   ├── modeling.ipynb            # 모델 구조·전처리 비교 및 선택 근거
│   ├── data/                     # 실험 데이터(Git 제외)
│   ├── checkpoints/              # 실험 가중치(Git 제외)
│   └── results.csv               # 실험 결과(Git 제외)
├── .gitignore
└── README.md
```

## 데이터셋 준비

모델링에는 Kaggle의 [WM-811K Wafer Map 데이터셋](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/data)을 사용합니다. 원본 데이터는 저장소에 포함하지 않으므로 모델링 노트북을 실행하기 전에 별도로 내려받아야 합니다.

다운로드한 압축 파일을 해제하고 `LSWMD.pkl`을 다음 위치에 저장합니다.

```text
MLOps_wafer/
└── notebooks/
    └── data/
        └── LSWMD.pkl
```

Kaggle CLI 설치와 API 인증이 완료되어 있다면 프로젝트 루트에서 다음 명령으로 받을 수 있습니다.

```powershell
New-Item -ItemType Directory -Force notebooks\data
kaggle datasets download -d qingyi/wm811k-wafer-map -p notebooks\data --unzip
```

다운로드 후 파일 경로가 `notebooks/data/LSWMD.pkl`인지 확인합니다. [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb)은 `data/LSWMD.pkl` 상대 경로를 사용하므로 다음과 같이 `notebooks`에서 Jupyter를 실행합니다.

```powershell
cd notebooks
jupyter notebook
```

`notebooks/data/`와 노트북에서 생성하는 `notebooks/checkpoints/`, `notebooks/results.csv`는 `.gitignore`에 등록되어 로컬에만 보관됩니다. 백엔드의 `back/data/`는 업로드된 평가·재학습 데이터를 저장하는 별도 경로이므로 원본 모델링 데이터는 `notebooks/data/`에 둡니다.

## 실행 환경

- Python 3.10 이상, 3.11 권장
- PostgreSQL 13 이상
- Node.js 18 이상 및 npm

## 로컬 실행 방법

### 1. 백엔드 환경 구성

프로젝트 루트에서 `back` 디렉터리로 이동한 후 가상환경을 구성합니다.

Windows PowerShell:

```powershell
cd back
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cd back
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

복사한 `.env`에서 OpenAI API 키와 PostgreSQL 접속 정보를 실제 환경에 맞게 수정합니다.

### 2. 데이터베이스 초기화

PostgreSQL 서버를 실행한 다음 아래 명령을 사용합니다.

```bash
python create_db.py
```

> 주의: `create_db.py`는 `.env`의 `DB_NAME`과 같은 데이터베이스가 존재하면 삭제하고 다시 생성합니다. 보존해야 하는 데이터베이스 이름을 사용하지 마세요.

### 3. 문헌 인덱스 생성

LLM 설명 기능을 사용하려면 OpenAI API 키를 설정한 후 문헌 인덱스를 생성합니다.

```bash
python -m scripts.build_paper_index
```

새 문헌이 추가되면 서비스가 인덱스를 주기적으로 갱신합니다.

### 4. 백엔드 실행

`back` 디렉터리에서 다음 명령을 실행합니다.

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

서버가 실행되면 [Swagger UI](http://localhost:8001/docs)에서 API를 확인할 수 있습니다.

### 5. 프런트엔드 실행

새 터미널에서 프로젝트의 `front` 디렉터리로 이동합니다.

```bash
cd front
npm install
npm run dev
```

브라우저에서 `http://localhost:5173`으로 접속합니다.

## 환경변수

| 변수 | 필수 여부 | 설명 | 기본값 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | LLM 기능 사용 시 | 임베딩 및 결함 설명에 사용하는 OpenAI API 키 | 없음 |
| `DB_NAME` | 필수 | PostgreSQL 데이터베이스 이름 | 없음 |
| `DB_USER` | 필수 | PostgreSQL 사용자 | 없음 |
| `DB_PASSWORD` | 필수 | PostgreSQL 비밀번호 | 없음 |
| `DB_HOST` | 필수 | PostgreSQL 호스트 | 없음 |
| `DB_PORT` | 필수 | PostgreSQL 포트 | 없음 |
| `DB_POOL_MIN` | 선택 | 최소 DB 연결 수 | `1` |
| `DB_POOL_MAX` | 선택 | 최대 DB 연결 수 | `5` |
| `DB_CONNECT_TIMEOUT` | 선택 | DB 연결 제한 시간(초) | `5` |
| `BASE_DIR` | 선택 | 백엔드 리소스의 기준 경로 | 현재 `back` 경로 |

## 재학습 데이터 형식

모델 평가 및 재학습 API는 pandas DataFrame을 저장한 pickle 파일을 입력으로 사용합니다.

| 컬럼 | 설명 |
| --- | --- |
| `waferMap` | 2차원 웨이퍼 맵 배열 |
| `failureType` | 9개 분류 클래스 중 하나인 정답 라벨 |
| `lotName` | 웨이퍼가 속한 Lot 식별자 |

업로드 데이터에서 현재 모델의 Macro F1이 `0.7` 미만이면 재학습을 추천하며, 실제 학습은 사용자가 화면의 재학습 버튼을 눌렀을 때 시작합니다. 재학습 후보는 다음 절차로 생성하고 승격합니다.

1. seed 42와 클래스 계층화를 사용해 데이터를 학습 80%, 검증 10%, 테스트 10%로 분할합니다.
2. 학습 데이터에만 클래스 평균 크기 오버샘플링과 좌우·상하 반전, ±15도 회전 증강을 적용합니다.
3. 2채널 CNN을 새 초기 가중치부터 학습하고 검증 Macro F1이 가장 높은 가중치를 저장합니다.
4. 후보와 현재 배포 모델을 동일한 테스트 데이터에서 9개 클래스 기준 Macro F1으로 평가합니다.
5. 후보 점수가 엄격하게 더 높을 때만 `best_model.pth`를 원자적으로 교체합니다. 학습·평가·파일 교체가 실패하거나 후보 점수가 낮으면 기존 모델을 유지합니다.

## 주요 API

| Method | Endpoint | 설명 |
| --- | --- | --- |
| `POST` | `/predict_img` | 단일 웨이퍼 이미지 예측 |
| `POST` | `/predict_multi_images_one_lot` | 단일 Lot의 다중 이미지 예측 |
| `POST` | `/predict_multi_images_multi_lots` | 여러 Lot의 다중 이미지 예측 |
| `POST` | `/upload_predict_labeled_images` | 라벨 데이터 업로드 및 기존 모델 평가 |
| `POST` | `/retrain_predict_labeled_images` | 모델 재학습, 성능 비교 및 모델 교체 |
| `POST` | `/explanation/get_llm_response` | 결함률과 결함 분포를 기반으로 원인 설명 생성 |

## 팀원별 역할

### 전혜민

- 프로젝트 기획 및 서비스 아키텍처 설계
- FastAPI 라우팅, 서비스 계층 및 DB 연동 구현
- AIOps 파이프라인 설계 및 구현
- OpenAI API 기반 LLM 응답 모듈 개발
- 데이터 전처리와 CNN 학습·추론 파이프라인 구축

### 김정윤

- 프로젝트 기획 및 서비스 아키텍처 설계
- Vue 기반 모델 관리 화면 구현
- Lot·Factory 관련 FastAPI API 설계 및 구현
- 모델 학습·추론 결과 시각화와 프런트엔드 API 연동
- 프런트엔드 UI/UX 및 상태 관리 개선

### 김유진

- 프로젝트 기획 및 서비스 아키텍처 설계
- 서비스 와이어프레임 설계
- Vue 기반 Image·Lot·Factory 화면 구현
- 이미지 및 공정 분석 API 연동

## 개선 계획

- 실행 환경별 API URL을 환경변수로 분리
- 운영 환경에 맞게 CORS 허용 범위 제한
- 데이터 및 모델 버전 관리 체계 도입
- 테스트 환경과 CI 파이프라인 구축
