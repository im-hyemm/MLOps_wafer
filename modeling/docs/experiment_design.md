# 웨이퍼 분류 실험 설계

## 목표와 데이터

작은 CNN 개선과 기하학 특징의 효과를 재현 가능한 순차 비교로 확인한다. 단일 실험 결과를 근거로 복잡한 모델이 반드시 우월하다고 가정하지 않는다. 모델, 학습법, 전처리 순으로 선택하므로 모든 조합의 전역 최적값을 보장하는 탐색은 아니다.

- 입력: LSWMD.pkl, 클래스는 Center/Donut/Edge-Loc/Edge-Ring/Loc/Random/Scratch/Near-full/none 순서.
- 빈 라벨은 제외하며 중첩 배열 라벨을 문자열로 정리한다. 라벨 `none`은 정상 클래스이므로 제거하지 않는다.
- 원본 `trainTestLabel`을 사용하지 않고 전체 라벨 데이터를 셔플·층화 80/10/10으로 재분할한다. 처음 20%를 분리하고 그 안을 절반으로 나눈다. 두 분할 모두 random_state=42, shuffle=True.
- 원본 행 위치 `source_index`와 전체 데이터 지문으로 분할 manifest를 저장한다. 모델 seed를 바꿔도 분할은 고정이다.
- 웨이퍼 단위 분할이므로 Lot 중복을 막지는 않는다. 중복 Lot 수를 기록하며 결과는 새 Lot 일반화 성능으로 해석하지 않는다.
- 이전 notebook의 81/9/10 및 서비스 분할과 달라 기존 점수와 직접 우열 비교하지 않는다.

## 모델과 학습 조건

모든 입력은 64×64이다. 기본 Adam lr=0.001, weight_decay=0.0001, batch_size=32, dropout은 아래 정의를 따른다. 최대 30 epoch, validation Macro-F1 기준 early stopping patience=7, ReduceLROnPlateau factor=0.5/patience=3. 최저 loss가 아니라 최고 validation Macro-F1 checkpoint를 복원한다. GPU에서는 AMP를 사용한다. seed와 deterministic 옵션을 기록하되 서로 다른 GPU·라이브러리 버전에서 비트 단위 동일 결과까지 보장하지 않는다.

| 모델 | 구조 |
|---|---|
| SmallCNN | Conv3×3+BN+ReLU, 32→64→128→256; 첫 3블록 MaxPool2; GAP1×1; Dropout .5; Linear256→9 |
| SpatialCNN | 동일 backbone, pooling2×2; Dropout .5; Linear1024→128+ReLU; Dropout .3; Linear128→9 |
| ResidualCNN | 32/64/128 단계마다 Conv+BN+ReLU 채널 변환, 동일 채널 2-Conv 잔차 블록, MaxPool2; pooling2×2; 512→128→9 MLP |
| HybridCNN | SpatialCNN 128차원 표현 + 특징24→32 ReLU; 연결160차원; Dropout .3; Linear160→9 |

SpatialCNN은 pooling뿐 아니라 head 파라미터 수도 달라지는 실험이다. ResidualCNN은 깊이·채널 구성도 달라지므로 잔차 연결만의 효과로 해석하지 않는다. 파라미터 수와 학습 시간을 함께 기록한다.

CE는 기본 CrossEntropyLoss다. Weighted CE는 train 클래스 개수 `n_c`로 `w_c = 1/sqrt(n_c)`를 계산해 클래스 간 평균 1로 정규화한다. sampler·SMOTE·다수 클래스 삭제는 적용하지 않는다. validation loss에도 같은 train 가중치를 사용하므로 CE와 Weighted CE의 loss 절대값은 직접 비교하지 않는다. 선택은 공통 Macro-F1으로 한다.

증강은 train에만 수평·수직 반전(각 p=.5), 각도 Uniform(-15,15) 회전, 최근접 보간·영역 밖 0으로 적용한다. 맵 개수는 늘리지 않으며 매 조회마다 무작위 변환한다. mask와 Hybrid 특징은 최종 변환 맵에서 계산한다. 회전에 따른 경계 clipping 가능성도 포함한 증강 정책의 효과를 비교한다.

## 총 11회 순차 실험

| 단계 | 조건 | 신규 학습 | 재사용 |
|---|---|---:|---|
| 1 모델 | 4모델 × Fixed/Weighted CE/증강 없음/seed42 | 4 | 없음 |
| 2 학습법 | 모델 우승자 × CE/Weighted CE × 증강 없음/있음 | 3 | Weighted CE/증강 없음 |
| 3 전처리 | 모델·학습법 우승자 × Fixed/Pad/Pad+Mask | 2 | Fixed |
| 4 seed | 최종 설정 × seed42/43/44 | 2 | seed42 |

각 단계의 우승 기준은 best validation Macro-F1 → worst-class F1 → 적은 파라미터 수다. 모두 같으면 실험 ID 정렬 순서로 결정한다. epoch checkpoint는 Macro-F1이 엄격히 증가할 때만 갱신하므로 같은 점수의 첫 epoch를 유지한다.

전처리는 최근접 보간을 사용한다. Fixed는 종횡비 무시, Pad는 종횡비 유지·중앙 zero padding, Pad+Mask는 Pad에 `(map > 0)` 채널을 추가한다. 맵 채널은 2로 나누며 mask는 0/1이다. 세 후보 모두 2단계에서 선택한 loss와 증강 조건을 동일하게 적용한다.

Test는 후보 9회에서 사용하지 않는다. 최종 설정을 JSON에 먼저 잠그고 seed43/44를 학습한 뒤, validation으로 배포 후보 seed를 선택하고 3개 모두 test 평가한다. 최종 보고는 3개 Macro-F1 평균·표본 표준편차(ddof=1)와 클래스별 지표다. test 성능으로 seed를 고르지 않는다. 11회는 완료 결과를 재사용하는 정상 실행 기준이며 실패·강제 재학습은 별도 실행이다.

## Hybrid 24개 특징 정의

특징은 리사이즈 및 증강 후 실제 이미지 입력의 범주 맵에서 계산한다. 원본 해상도 특징과 모델 입력의 불일치를 피하기 위한 선택이다. Lot ID와 정답 라벨은 특징에 포함하지 않는다.

| 번호 | 특징 | 정의 |
|---|---|---|
| 0–3 | has_defect, defect_ratio, valid_area_ratio, wafer_aspect_ratio | 불량 존재, 불량/유효 셀, 유효 셀/전체 픽셀, 유효 bbox 가로/세로 |
| 4–7 | centroid_x/y, radius_mean/std | 유효 bbox 중심 기준 반폭·반높이 정규화 좌표; 유효 영역 최대 반경으로 나눈 불량 반경 통계 |
| 8–10 | radial_density_0/1/2 | 정규화 반경 [0,1/3), [1/3,2/3), [2/3,1] 구역의 불량/유효 셀 |
| 11–18 | sector_density_0..7 | atan2 각도 [-π,π]를 8등분한 구역의 불량/유효 셀 |
| 19–21 | component_count, largest_component_ratio, eccentricity | 8연결 불량 영역 수, 최대 영역/전체 불량, 불량 좌표 covariance 고유값으로 sqrt(1−λmin/λmax) |
| 22–23 | orientation_cos2/sin2 | covariance 주축 방향 θ의 cos(2θ), sin(2θ) |

빈 구역 밀도는 0이다. 결함이 없으면 웨이퍼 면적·종횡비 외 특징은 0이다. 결함 1개 또는 분산 0이면 shape 통계는 0, 등방성 분포이면 방향은 0이다. 변환 후 유효 셀이 없으면 모든 특징은 0이다. train의 증강 없는 맵에서 평균·표준편차를 fit한다. std가 1e-6 미만인 상수 특징은 배율 1을 사용해 증강 후 값이 폭증하지 않게 한다. 해당 통계만 validation/test/증강 train에 적용한다.

## 진행 상황과 보존

각 epoch에 train/validation의 loss, accuracy, macro·micro·weighted precision/recall/F1, worst-class F1, 전체 support와 9개 클래스별 precision/recall/F1/support를 `history.csv`에 저장한다. `history_details.json`은 동일 지표와 클래스 순서, epoch별 원시 혼동행렬을 기록한다. support는 실제 관측 샘플 수이며 weighted 평균은 support로 가중한다. 단일 정답 9클래스 문제에서 micro precision/recall/F1은 accuracy와 같다. 누락 클래스의 지표는 0으로 처리하고 macro 평균은 고정 9클래스를 사용한다. 추가 forward pass 없이 기존 epoch 예측을 집계하며 test 데이터는 사용하지 않는다.

train 지표는 학습 중 증강·dropout과 갱신 중인 가중치의 영향을 포함한다. validation 지표는 해당 epoch 종료 시 평가 모드에서 계산한다. 원시 혼동행렬에서 후속 정규화와 클래스 오류 분석이 가능하지만, epoch별 확률은 저장하지 않으므로 임의의 ROC-AUC나 임계값 분석용 기록은 아니다. 예전 학습에서 누락한 epoch 지표는 소급 복원하지 않는다.

```text
단계=models 후보=2/4 고유 학습 완료=1/11
전처리·특징 준비 ...
spatial_cnn_... epoch 5 train: 850/4324 [경과<ETA, batch/s, loss=...]
Epoch 5/30 train loss=0.35 F1=0.82 | val loss=0.40 F1=0.79 | lr=0.001 90초 best=0.81 wait=2/7
최고 checkpoint 저장 epoch=8 F1=0.85
고유 학습 완료=2/11
```

상태는 running/completed/failed/interrupted다. phase는 starting/preparing/train/validation/epoch_completed/saving_validation/completed다. status.json은 30초 간격 또는 epoch·예외·완료 시 즉시 갱신한다. 매우 긴 단일 배치·파일 I/O가 멈추면 그 동안 heartbeat도 갱신되지 않을 수 있다. 전체 남은 시간은 early stopping 때문에 확정값을 표시하지 않는다.

train.log는 주요 이벤트와 traceback을 남기며 배치마다 파일에 로그를 쓰지 않는다. history.csv는 매 epoch 저장한다. 예외·사용자 중단은 상태에 기록하고 재실행은 처음부터 학습한다. 런타임 강제 종료는 기록할 수 없으므로 마지막 갱신 시각을 확인한다. 모델·JSON·history는 같은 폴더의 임시 파일에 쓴 뒤 교체한다.

설정·데이터 분할·Python 소스가 같은 완료 실험만 재사용한다. 코드가 변경되면 새 suite와 실험 ID를 사용한다. notebook은 실행과 시각화만 담당하며 학습 함수 정의를 복사하지 않는다. 로그 조회를 위해 같은 실행 중 커널에 새 셀을 제출하면 대기하므로, 학습 셀 또는 Drive 파일을 확인한다.
