# 웨이퍼 모델링 실험

`compare_experiments.ipynb`에서 Colab 학습과 시각화를 실행한다. 학습 구현은 Python 모듈에 있으며 CLI도 같은 함수를 호출한다. 기존 `notebooks/`와 `back/`의 실험·서비스 코드와 독립적으로 사용한다.

## Colab / VS Code Colab extension

1. 이 `modeling` 폴더를 Google Drive의 `MyDrive/Colab Notebooks/SKALA/wafer_v2/modeling`에 복사한다.
2. 데이터는 같은 프로젝트의 `data/LSWMD.pkl`에 둔다.
3. `compare_experiments.ipynb`를 열고 Colab GPU 커널을 선택한다.
4. 환경·데이터 셀, 모델 비교, 학습법 비교, 전처리 비교, 최종 검증 셀을 순서대로 실행한다.

로컬 notebook 파일을 열어도 Python 실행과 파일 접근은 원격 Colab 커널에서 일어난다. 로컬 `.py` 수정은 Drive에 직접 복사해야 반영된다. 코드를 바꿨으면 커널을 다시 시작한다. 소스 해시가 바뀌면 이전 실험과 재사용·혼합 비교하지 않는다.

기본 저장 위치는 `/content/drive/MyDrive/Colab Notebooks/SKALA/wafer_v2/modeling/artifacts`이다. 별도 다운로드 자동화는 없으며 사용자가 필요한 폴더를 Drive에서 로컬로 내려받는다.

## Python API

```python
from modeling.config import (
    ExperimentConfig,
    DEFAULT_DATA_PATH,
    DEFAULT_ARTIFACT_ROOT,
)
from modeling.data import load_dataset, create_or_load_splits
from modeling.training import configure_logging, train_experiment
from modeling.run_experiment import run_stage
from modeling.evaluation import get_experiment_status, load_experiment_results

configure_logging()
splits = create_or_load_splits(
    load_dataset(DEFAULT_DATA_PATH), DEFAULT_ARTIFACT_ROOT
)
config = ExperimentConfig(epochs=30, batch_size=32)
run_stage("models", splits, DEFAULT_ARTIFACT_ROOT, config)
run_stage("recipes", splits, DEFAULT_ARTIFACT_ROOT, config)
run_stage("preprocessing", splits, DEFAULT_ARTIFACT_ROOT, config)
# 최종 설정을 고정한 후 test 평가까지 실행한다.
run_stage("final", splits, DEFAULT_ARTIFACT_ROOT, config)
display(get_experiment_status(DEFAULT_ARTIFACT_ROOT))
```

`run_stage("all", ...)`은 네 단계를 순서대로 실행한다. `train_experiment(...)`는 단일 설정을 학습하며 test는 평가하지 않는다. 단계 실행은 동일 실험을 자동 재사용한다. 단일 실험의 `force=True`는 해당 실험을 처음부터 다시 학습하고 이전 test 산출물을 제거한다. 이미 만든 suite 보고서는 과거 보고서이므로 강제 재학습 후에는 `final` 단계를 다시 실행해 갱신한다.

## CLI

프로젝트 루트에서 실행한다. Colab에서는 셀에서 `!python -m ...`로 호출할 수도 있다.

```bash
python -m modeling.run_experiment --stage models
python -m modeling.run_experiment --stage recipes
python -m modeling.run_experiment --stage preprocessing
python -m modeling.run_experiment --stage final
python -m modeling.run_experiment --model spatial_cnn --loss ce --augmentation
```

`--data-path`, `--artifact-root`, `--epochs`, `--batch-size`, `--device`로 환경을 바꾼다. `--force`는 단일 실험에서만 지원한다. 전체 후보·단계 실행은 train seed 42를 고정하며 최종 단계만 43·44를 추가한다.

## 진행 로그와 재시작

- 화면: 단계별 완료 수 `/11`, 준비 진행률, train/validation 배치 수·loss·속도·경과 시간·현재 epoch ETA, epoch 요약.
- `train.log`: 설정·장치·클래스 분포·파라미터 수, epoch 결과, checkpoint·학습률 변경, 완료·재사용·중단·traceback. 재실행 시 append한다.
- `status.json`: 단계, 상태, epoch, 배치, UTC 갱신 시각, 경과 시간. 최대 30초 간격이며 주요 이벤트는 즉시 갱신한다.
- `history.csv`: epoch 종료마다 원자적으로 갱신한다.
- `suites/<ID>/status.json`: 해당 11회 실험 흐름의 단계와 완료 ID 목록.

`get_experiment_status(root)`의 `seconds_since_update`로 heartbeat가 멈췄는지 확인할 수 있다. 같은 notebook 커널이 학습 중이면 다른 셀은 실행을 기다리므로 실시간 확인은 학습 셀 출력 또는 Drive 파일을 사용한다. 강제 런타임 종료는 마지막 상태가 running으로 남을 수 있다. 중단 실험은 epoch 중간 재개 없이 처음부터 다시 학습한다.

완료 재사용은 설정·전체 데이터/분할 해시·Python 소스 해시와 필수 산출물을 확인한다. 따라서 수정한 코드로 이전 checkpoint를 조용히 재사용하지 않는다. 전처리와 특징은 분할별로 메모리에 준비하며 학습 시 증강만 온라인으로 수행한다.

## 산출물

`history.csv`는 매 epoch의 train/validation loss, accuracy, macro·micro·weighted precision/recall/F1, worst-class F1, 전체 support와 9개 클래스별 precision/recall/F1/support를 모두 저장한다. 기존 컬럼 이름은 유지한다. 클래스별 컬럼 예시는 `validation_Scratch_recall`, `train_none_precision`이다.

`history_details.json`에는 같은 지표와 epoch별 train/validation 9×9 confusion matrix를 저장한다. `classes`에 행·열 순서가 있으며 행은 정답, 열은 예측이다. 추가 추론 없이 학습 중 수집한 정답·예측으로 계산한다. train 지표는 증강·dropout이 활성화된 학습 중 여러 가중치 상태에서 얻은 예측의 집계이며, epoch 종료 모델의 재평가 점수는 아니다.

이 변경은 앞으로 실행하는 학습에 적용된다. 기존 history에 저장하지 않은 과거 epoch 지표를 소급 복원하지 않는다. epoch별 확률과 모든 epoch의 모델 가중치는 저장하지 않으므로 ROC-AUC·임계값 분석까지 포함한 임의의 지표를 복원하는 형식은 아니다. Python 소스 변경으로 새 실험 ID가 생성되며 기존 결과 파일은 유지된다.

각 실험에 `config.json`, `metrics.json`, `history.csv`, `predictions.csv`, `confusion_matrix.png`, `model.pth`, `train.log`, `status.json`이 저장된다. 기본 예측·그림·지표는 validation 기준이다. 최종 평가에는 `test_metrics.json`, `test_predictions.csv`, `test_confusion_matrix.png`가 추가된다.

최종 집계는 `suites/<ID>/final_summary.json`에 저장하며 3-seed test 평균과 표본 표준편차(ddof=1)를 제공한다. `deployment_selection.json`은 test를 보기 전에 validation으로 고른 seed를 가리킨다. checkpoint는 모델 가중치, 설정, 클래스·특징 순서, train 특징 정규화 통계, 분할·코드 ID를 포함한다. 서비스에 자동 배포하지 않는다.

## 로컬 검증

GPU가 없어도 짧은 합성 데이터 테스트와 결과 조회가 가능하다. Python 3.11 이상을 사용한다.

```powershell
python -m venv .venv-modeling
.\.venv-modeling\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-modeling\Scripts\python -m pip install -r modeling/requirements.txt pytest nbformat
.\.venv-modeling\Scripts\python -m pytest modeling/tests -q
```

설계와 특징 정의는 [실험 설계](docs/experiment_design.md)를 참고한다.
