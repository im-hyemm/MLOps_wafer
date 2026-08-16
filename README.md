# WaferGuard 프로젝트 README (배포·이관용)

본 문서는 다른 환경(상대방 PC)에서 **프로젝트가 문제없이 실행**되도록 하기 위한 단계별 안내서입니다. 운영체제는 Windows 10/11 및 macOS/Linux를 가정합니다. 경로 표기는 예시이므로 각 환경에 맞게 수정하십시오.

---

## 1) 프로젝트 개요

- **back/**: FastAPI 기반 백엔드 (DB 연동, 모델 추론/학습 API)
- **front/**: Vue 3 + Vite 기반 프론트엔드 (대시보드/업로드 UI)
- **presentation_data/**: 발표(데모)에 사용한 정적 데이터
- **training_data/**: CNN 학습용 이미지·라벨 데이터

> 참고: 실제 구동에 필요한 코드는 `back/`, `front/`에 있으며, `presentation_data/`와 `training_data/`는 시연/학습 목적의 데이터 폴더입니다.

---

## 2) 사전 요구사항 (Prerequisites)

- **Python** 3.10+ (권장: 3.11)
- **PostgreSQL** 13+ (로컬 실행 가정)
- **Node.js** 18+ 및 **npm** (프런트엔드용)
- (Windows 권장) PowerShell, (macOS/Linux) Bash/Zsh

---

## 3) 백엔드 설정 (back/)

### 3.1. 환경 변수(.env) 작성

`back/` 디렉터리 **최상단**에 `.env` 파일을 생성하고 아래 형식을 본인 환경에 맞게 채우십시오.

```ini
# 파일 위치: back/.env
# 경로는 본인 환경에 맞게 수정하세요
BASE_DIR=C:/Users/<user>/back/

# OpenAI API 키 (예: sk-...) — 실제 키로 교체
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# PostgreSQL 연결 정보
DB_NAME=YOUR_DB_NAME
DB_USER=postgres
DB_PASSWORD=YOUR_DB_PASSWORD
DB_HOST=YOUR_DB_HOST
DB_PORT=YOUR_DB_PORT

# 연결 풀 설정 (필요시 조정)
DB_POOL_MIN=1
DB_POOL_MAX=5
DB_CONNECT_TIMEOUT=5
```

> **보안 주의**: 공유 또는 공개 저장소에 `.env`를 업로드하지 마십시오.

### 3.2. 가상환경 생성 및 의존성 설치

아래 명령은 모두 `back/` 디렉터리에서 실행합니다.

**Windows (PowerShell)**

```powershell
cd back
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**

```bash
cd back
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. 데이터베이스 초기화

PostgreSQL 서버가 실행 중인지 확인한 뒤, 아래 스크립트를 실행하여 DB 및 스키마를 준비합니다.

```bash
# 위치: back/
python create_db.py
```

> 기본 연결 정보는 `.env`를 따릅니다. 필요 시 `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`를 조정하십시오.

### 3.4. 백엔드 실행

개발 서버(예: Uvicorn) 실행 예시는 다음과 같습니다.

```bash
# 위치: back/
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

서버가 정상 기동되면 `http://localhost:8001/docs` (Swagger UI)에서 API를 확인할 수 있습니다.

---

## 4) 프런트엔드 설정 (front/)

### 4.1. 의존성 설치

아래 명령은 모두 `front/` 디렉터리에서 실행합니다.

```bash
cd front
npm install chart.js chartjs-plugin-datalabels axios vue-router bootstrap-vue-next bootstrap
```

> Vue 3 및 Vite는 프로젝트에 이미 설정되어 있다고 가정합니다. 필요 시 `npm install` 만으로 루트의 `package.json`에 정의된 기본 의존성도 함께 설치하십시오.

### 4.2. 프런트엔드 실행

```bash
# 위치: front/
npm run dev
```

기본적으로 Vite 개발 서버는 `http://localhost:5173`에서 실행됩니다. 프록시 설정 여부에 따라 백엔드(`http://localhost:8001`)와 통신합니다.

---

## 5) 학습/발표 라우터 전환 안내

현재 기본 라우터는 **`training.py`**로 설정되어 있으며, 그대로 실행하면 최대 **에폭 100회** 학습이 동작합니다. **발표(데모)와 동일한 동작**을 원하시면 `main.py`에서 라우터 임포트를 **`training_presentation.py`**로 변경하십시오.

### 5.1. 기본

```python
# 파일: back/main.py
from api.routes.training import router as training_router
app.include_router(training_router, prefix="/training")
```

### 5.2. 발표 모드

```python
# 파일: back/main.py
from api.routes.training_presentation import router as training_router
app.include_router(training_router, prefix="/training")
```

---

## 6) 실행 순서 요약 (Cheat Sheet)

1. **PostgreSQL 설치 및 실행**
2. `back/`로 이동 → **`.env` 작성**
3. `back/`에서 **가상환경 생성** 및 `pip install -r requirements.txt`
4. `back/create_db.py` 실행 (스키마/초기화)
5. 필요 시 `main.py` 라우터를 `training_presentation`으로 전환 (발표 모드)
6. `python -m uvicorn main:app --port 8001 --reload` 로 백엔드 실행
7. `front/`로 이동 → `npm install ...` 설치 → `npm run dev` 실행

---

## 7) 폴더 구조

```
project-root/
├─ back/                # FastAPI backend and ML pipeline core
│  ├─ api/              # HTTP endpoints and route handlers
│  │  └─ routes/        # Training & presentation APIs
│  ├─ config/           # Shared config (DB paths, settings)
│  ├─ data/             # Dataset loading utilities
│  ├─ evaluation/       # Metrics & visualization helpers
│  ├─ model/            # Model architecture, trainer, predictor
│  ├─ scripts/          # Support scripts (e.g., paper index builder)
│  └─ services/         # Reusable services (DB access, LLM client)
├─ front/               # Vite/Vue frontend application
│  ├─ src/              # Vue components, pages, router, assets
│  └─ package.json      # Frontend dependencies & scripts
├─ presentation_data/   # Demo assets for presentations
└─ training_data/       # Core training datasets

```

# WaferGuard 프로젝트 README (배포·이관용)

> 작성자: SKALA 2기 4반 전혜민, 김정윤, 김유진

본 문서는 다른 환경(상대방 PC)에서 **프로젝트가 문제없이 실행**되도록 하기 위한 단계별 안내서입니다. 운영체제는 Windows 10/11 및 macOS/Linux를 가정합니다. 경로 표기는 예시이므로 각 환경에 맞게 수정하십시오.

---

## 1) 프로젝트 개요

- **back/**: FastAPI 기반 백엔드 (DB 연동, 모델 추론/학습 API)
- **front/**: Vue 3 + Vite 기반 프론트엔드 (대시보드/업로드 UI)
- **presentation_data/**: 발표(데모)에 사용한 정적 데이터
- **training_data/**: CNN 학습용 이미지·라벨 데이터

> 참고: 실제 구동에 필요한 코드는 `back/`, `front/`에 있으며, `presentation_data/`와 `training_data/`는 시연/학습 목적의 데이터 폴더입니다.

---

## 2) 사전 요구사항 (Prerequisites)

- **Python** 3.10+ (권장: 3.11)
- **PostgreSQL** 13+ (로컬 실행 가정)
- **Node.js** 18+ 및 **npm** (프런트엔드용)
- (Windows 권장) PowerShell, (macOS/Linux) Bash/Zsh

---

## 3) 백엔드 설정 (back/)

### 3.1. 환경 변수(.env) 작성

`back/` 디렉터리 **최상단**에 `.env` 파일을 생성하고 아래 형식을 본인 환경에 맞게 채우십시오.

```ini
# 파일 위치: back/.env
# 경로는 본인 환경에 맞게 수정하세요
BASE_DIR=C:/Users/<user>/back/

# OpenAI API 키 (예: sk-...) — 실제 키로 교체
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# PostgreSQL 연결 정보
DB_NAME=YOUR_DB_NAME
DB_USER=postgres
DB_PASSWORD=YOUR_DB_PASSWORD
DB_HOST=YOUR_DB_HOST
DB_PORT=YOUR_DB_PORT

# 연결 풀 설정 (필요시 조정)
DB_POOL_MIN=1
DB_POOL_MAX=5
DB_CONNECT_TIMEOUT=5
```

> **보안 주의**: 공유 또는 공개 저장소에 `.env`를 업로드하지 마십시오.

### 3.2. 가상환경 생성 및 의존성 설치

아래 명령은 모두 `back/` 디렉터리에서 실행합니다.

**Windows (PowerShell)**

```powershell
cd back
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**

```bash
cd back
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. 데이터베이스 초기화

PostgreSQL 서버가 실행 중인지 확인한 뒤, 아래 스크립트를 실행하여 DB 및 스키마를 준비합니다.

```bash
# 위치: back/
python create_db.py
```

> 기본 연결 정보는 `.env`를 따릅니다. 필요 시 `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`를 조정하십시오.

### 3.4. 백엔드 실행

개발 서버(예: Uvicorn) 실행 예시는 다음과 같습니다.

```bash
# 위치: back/
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

서버가 정상 기동되면 `http://localhost:8001/docs` (Swagger UI)에서 API를 확인할 수 있습니다.

---

## 4) 프런트엔드 설정 (front/)

### 4.1. 의존성 설치

아래 명령은 모두 `front/` 디렉터리에서 실행합니다.

```bash
cd front
npm install chart.js chartjs-plugin-datalabels axios vue-router bootstrap-vue-next bootstrap
```

> Vue 3 및 Vite는 프로젝트에 이미 설정되어 있다고 가정합니다. 필요 시 `npm install` 만으로 루트의 `package.json`에 정의된 기본 의존성도 함께 설치하십시오.

### 4.2. 프런트엔드 실행

```bash
# 위치: front/
npm run dev
```

기본적으로 Vite 개발 서버는 `http://localhost:5173`에서 실행됩니다. 프록시 설정 여부에 따라 백엔드(`http://localhost:8001`)와 통신합니다.

---

## 5) 학습/발표 라우터 전환 안내

현재 기본 라우터는 **`training.py`**로 설정되어 있으며, 그대로 실행하면 최대 **에폭 100회** 학습이 동작합니다. **발표(데모)와 동일한 동작**을 원하시면 `main.py`에서 라우터 임포트를 **`training_presentation.py`**로 변경하십시오.

### 5.1. 기본

```python
# 파일: back/main.py
from api.routes.training import router as training_router
app.include_router(training_router, prefix="/training")
```

### 5.2. 발표 모드

```python
# 파일: back/main.py
from api.routes.training_presentation import router as training_router
app.include_router(training_router, prefix="/training")
```

---

## 6) 실행 순서 요약 (Cheat Sheet)

1. **PostgreSQL 설치 및 실행**
2. `back/`로 이동 → **`.env` 작성**
3. `back/`에서 **가상환경 생성** 및 `pip install -r requirements.txt`
4. `back/create_db.py` 실행 (스키마/초기화)
5. 필요 시 `main.py` 라우터를 `training_presentation`으로 전환 (발표 모드)
6. `python -m uvicorn main:app --port 8001 --reload` 로 백엔드 실행
7. `front/`로 이동 → `npm install ...` 설치 → `npm run dev` 실행

---

## Contributors

- 전혜민 (SKALA 2기 4반)

  - 프로젝트 기획 및 서비스 아키텍처 설계
  - FastAPI 백엔드 전반 설계 및 구현 (라우팅, 서비스 계층, DB 연동)
  - AIOps 파이프라인 설계 및 구현
  - LLM 응답 요청 모듈 개발 (OpenAI API 연동)
  - 데이터 전처리 및 CNN 기반 모델 학습·추론 파이프라인 구축

- 김정윤 (SKALA 2기 4반)

  - 프로젝트 기획 및 서비스 아키텍처 설계
  - Vue.js 기반 모델 관리(Model Management) 탭 프론트엔드 구현
  - Lot·Factory 탭 관련 FastAPI 라우터 및 API 설계·구현
  - 프론트엔드와 백엔드 API 연동 (모델 학습/추론 결과 시각화, 관리 기능 포함)
  - 프론트엔드 UI/UX 개선 및 상태 관리 로직 구현

- 김유진 (SKALA 2기 4반)
  - 프로젝트 기획 및 서비스 아키텍처 설계
  - 서비스 와이어프레임 설계
  - Vue.js 기반 이미지(Image)·로트(Lot)·공장(Factory) 탭 프론트엔드 구현
  - 프론트엔드와 백엔드 API 연동 (모델 학습/추론 결과 시각화, 관리 기능 포함)
