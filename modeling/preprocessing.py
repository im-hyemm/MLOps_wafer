"""범주형 웨이퍼 맵의 전처리와 온라인 증강."""

import cv2
import numpy as np


def preprocess_map(wafer_map, mode="fixed_resize", target_size=64):
    """최근접 리사이즈를 적용한 정사각형 범주 맵을 반환한다.

    Args:
        wafer_map: 0=영역 밖, 1=정상, 2=불량인 2차원 배열.
        mode: fixed_resize, resize_pad 또는 resize_pad_mask.
        target_size: 출력 한 변의 크기.

    Returns:
        uint8 정사각형 웨이퍼 맵.

    Raises:
        ValueError: 입력 범주, 차원 또는 모드가 잘못된 경우.
    """
    source = np.asarray(wafer_map)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("비어 있지 않은 2차원 웨이퍼 맵이 필요합니다.")
    if not np.isin(source, (0, 1, 2)).all() or not np.any(source > 0):
        raise ValueError(
            "웨이퍼 맵은 0/1/2로 구성되고 유효 셀이 있어야 합니다."
        )
    source = source.astype(np.uint8)
    if mode == "fixed_resize":
        return cv2.resize(
            source, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )
    if mode not in ("resize_pad", "resize_pad_mask"):
        raise ValueError(f"지원하지 않는 전처리: {mode}")
    height, width = source.shape
    scale = target_size / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(
        source, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )
    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized
    return padded


def augment_map(wafer_map):
    """수평·수직 반전과 ±15도 최근접 회전을 온라인으로 적용한다.

    Args:
        wafer_map: 리사이즈된 0/1/2 범주 맵.

    Returns:
        반전과 회전 후 범주 맵.
    """
    result = wafer_map
    if np.random.random() < 0.5:
        result = np.fliplr(result)
    if np.random.random() < 0.5:
        result = np.flipud(result)
    height, width = result.shape
    transform = cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2),
        np.random.uniform(-15, 15),
        1.0,
    )
    return cv2.warpAffine(
        np.ascontiguousarray(result),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def encode_map(wafer_map, mode):
    """맵과 실제 셀 mask를 채널 우선 float32 입력으로 변환한다.

    Args:
        wafer_map: 전처리·증강 후 범주 맵.
        mode: 전처리 모드. resize_pad_mask만 두 채널을 사용한다.

    Returns:
        (채널, 높이, 너비) 형태의 float32 배열.
    """
    channels = [wafer_map.astype(np.float32) / 2.0]
    if mode == "resize_pad_mask":
        channels.append((wafer_map > 0).astype(np.float32))
    return np.ascontiguousarray(np.stack(channels))
