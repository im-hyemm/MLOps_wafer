"""Hybrid 모델의 24개 기하학 특징."""

import cv2
import numpy as np

FEATURE_NAMES = (
    "has_defect",
    "defect_ratio",
    "valid_area_ratio",
    "wafer_aspect_ratio",
    "centroid_x",
    "centroid_y",
    "radius_mean",
    "radius_std",
    "radial_density_0",
    "radial_density_1",
    "radial_density_2",
    *tuple(f"sector_density_{i}" for i in range(8)),
    "component_count",
    "largest_component_ratio",
    "eccentricity",
    "orientation_cos2",
    "orientation_sin2",
)


def extract_geometric_features(wafer_map):
    """실제 모델 입력과 같은 범주 맵에서 24개 특징을 계산한다.

    Args:
        wafer_map: 증강까지 적용된 2차원 0/1/2 범주 맵.

    Returns:
        FEATURE_NAMES 순서의 유한한 float32 벡터.
    """
    valid = wafer_map > 0
    defects = wafer_map == 2
    result = np.zeros(24, dtype=np.float32)
    vy, vx = np.where(valid)
    if len(vx) == 0:
        return result
    width, height = vx.max() - vx.min() + 1, vy.max() - vy.min() + 1
    result[2:4] = valid.mean(), width / height
    if not defects.any():
        return result
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
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        defects.astype(np.uint8), connectivity=8
    )
    result[19] = count - 1
    result[20] = stats[1:, cv2.CC_STAT_AREA].max() / defects.sum()
    if defects.sum() > 1:
        covariance = np.cov(np.stack((xx[defects], yy[defects])))
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] > 1e-10:
            result[21] = np.sqrt(max(0, 1 - values[0] / values[-1]))
            # 등방성 분포에는 방향을 부여하지 않는다.
            if values[-1] - values[0] > 1e-10:
                angle = np.arctan2(vectors[1, -1], vectors[0, -1])
                result[22:24] = np.cos(2 * angle), np.sin(2 * angle)
    return result
