"""Hybrid 모델의 24개 기하학 특징."""

import cv2
import numpy as np

FEATURE_NAMES = (
    "has_defect",  # 불량 셀 존재 여부(없음=0, 있음=1)
    "defect_ratio",  # 유효 웨이퍼 셀 중 불량 셀의 비율
    "valid_area_ratio",  # 전체 맵에서 유효 웨이퍼 셀이 차지하는 비율
    "wafer_aspect_ratio",  # 유효 웨이퍼 영역의 너비/높이 비율
    "centroid_x",  # 웨이퍼 중심을 기준으로 정규화한 불량 중심의 x 좌표
    "centroid_y",  # 웨이퍼 중심을 기준으로 정규화한 불량 중심의 y 좌표
    "radius_mean",  # 불량 셀의 정규화 반경 평균
    "radius_std",  # 불량 셀의 정규화 반경 표준편차
    "radial_density_0",  # 중심부에서 유효 셀 대비 불량 셀의 비율
    "radial_density_1",  # 중간부에서 유효 셀 대비 불량 셀의 비율
    "radial_density_2",  # 외곽부에서 유효 셀 대비 불량 셀의 비율
    # 웨이퍼를 각도 기준 8개 구역으로 나눈 각 구역의 불량 밀도
    *tuple(f"sector_density_{i}" for i in range(8)),
    "component_count",  # 서로 연결된 불량 영역의 개수
    "largest_component_ratio",  # 전체 불량 중 가장 큰 연결 영역의 비율
    "eccentricity",  # 불량 분포가 한 방향으로 늘어난 정도
    "orientation_cos2",  # 불량 분포 주축 방향의 cos(2θ)
    "orientation_sin2",  # 불량 분포 주축 방향의 sin(2θ)
)


def extract_geometric_features(wafer_map):
    """실제 모델 입력과 같은 범주 맵에서 24개 특징을 계산한다.

    Args:
        wafer_map: 증강까지 적용된 2차원 0/1/2 범주 맵.

    Returns:
        FEATURE_NAMES 순서의 유한한 float32 벡터.
    """
    valid = wafer_map > 0  # 정상과 불량을 포함한 실제 웨이퍼 영역
    defects = wafer_map == 2  # 불량 셀 영역
    result = np.zeros(24, dtype=np.float32)  # FEATURE_NAMES 순서의 특징 벡터

    # 유효 영역의 좌표와 경계 상자 크기를 구한다.
    vy, vx = np.where(valid)
    if len(vx) == 0:
        return result
    width, height = vx.max() - vx.min() + 1, vy.max() - vy.min() + 1
    result[2:4] = valid.mean(), width / height
    if not defects.any():
        return result
    # 좌표를 웨이퍼 중심 기준으로 옮기고 반지름 크기로 정규화한다.
    yy, xx = np.indices(wafer_map.shape)
    xx = (xx - (vx.max() + vx.min()) / 2) / max(width / 2, 1)
    yy = (yy - (vy.max() + vy.min()) / 2) / max(height / 2, 1)
    radius = np.sqrt(xx**2 + yy**2)
    radius /= max(float(radius[valid].max()), 1e-8)
    result[:2] = 1, defects.sum() / valid.sum()
    result[4:8] = (
        xx[defects].mean(),
        yy[defects].mean(),
        radius[defects].mean(),
        radius[defects].std(),
    )
    # 반경은 3개 구간, 각도는 8개 구역으로 이산화한다.
    radial_bins = np.minimum((radius * 3).astype(int), 2)
    sectors = np.minimum(
        ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * 8).astype(int), 7
    )
    for offset, bins, count in ((8, radial_bins, 3), (11, sectors, 8)):
        for index in range(count):
            region = valid & (bins == index)
            result[offset + index] = (defects & region).sum() / max(
                region.sum(), 1
            )
    # 8방향으로 인접한 불량 셀들을 하나의 연결 영역으로 묶는다.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        defects.astype(np.uint8), connectivity=8
    )
    result[19] = count - 1
    result[20] = stats[1:, cv2.CC_STAT_AREA].max() / defects.sum()
    if defects.sum() > 1:
        # 공분산 행렬의 고유값과 고유벡터로 모양과 주축 방향을 계산한다.
        covariance = np.cov(np.stack((xx[defects], yy[defects])))
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] > 1e-10:
            result[21] = np.sqrt(max(0, 1 - values[0] / values[-1]))
            # 등방성 분포에는 방향을 부여하지 않는다.
            if values[-1] - values[0] > 1e-10:
                angle = np.arctan2(vectors[1, -1], vectors[0, -1])
                result[22:24] = np.cos(2 * angle), np.sin(2 * angle)
    return result
