from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config.settings import TARGET_SIZE


def resize_and_pad(image, target_size=TARGET_SIZE):
    """이미지 비율을 유지해 리사이즈하고 패딩합니다.

    Args:
        image: 2차원 웨이퍼 맵 배열입니다.
        target_size: 목표 높이와 너비입니다.

    Returns:
        패딩된 2차원 ``uint8`` 배열입니다.
    """
    image = np.asarray(image)
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError("웨이퍼 맵은 비어 있지 않은 2차원 배열이어야 합니다.")

    target_h, target_w = target_size
    height, width = image.shape
    scale = min(target_h / height, target_w / width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    resized_image = cv2.resize(
        image.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    padded_image = np.zeros(target_size, dtype=np.uint8)
    top = (target_h - new_height) // 2
    left = (target_w - new_width) // 2
    padded_image[
        top:top + new_height,
        left:left + new_width,
    ] = resized_image
    return padded_image


def preprocess_wafer_map(
    image,
    resize_mode="resize_pad",
    target_size=TARGET_SIZE,
):
    """모델 입력 사양에 맞는 채널 우선 배열을 생성합니다.

    Args:
        image: 2차원 웨이퍼 맵 배열입니다.
        resize_mode: ``resize_pad`` 전처리 방식입니다.
        target_size: 목표 높이와 너비입니다.

    Returns:
        정규화된 ``(채널, 높이, 너비)`` float32 배열입니다.

    Raises:
        ValueError: 지원하지 않는 전처리 모드인 경우 발생합니다.
    """
    if resize_mode != "resize_pad":
        raise ValueError(f"지원하지 않는 전처리 모드입니다: {resize_mode}")

    source = np.asarray(image)
    if not np.isin(source, (0, 1, 2)).all() or not np.any(source > 0):
        raise ValueError(
            "웨이퍼 맵은 0/1/2로 구성되고 유효 셀이 있어야 합니다."
        )
    padded = resize_and_pad(source, target_size)
    return padded.astype(np.float32)[None, ...] / 2.0


def convert_into_colored_img(image_bytes_io):
    """웨이퍼 맵 이미지를 RGBA 이미지로 변환합니다.

    Args:
        image_bytes_io: 원본 이미지 바이트 스트림입니다.

    Returns:
        PNG 형식의 메모리 버퍼입니다.
    """
    image = Image.open(image_bytes_io)
    image = np.array(image)
    h, w = image.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    rgba[image == 0] = (0, 0, 0, 0)
    rgba[image == 1] = (192, 192, 192, 255)
    rgba[image == 2] = (255, 0, 0, 255)
    
    colored_img = Image.fromarray(rgba)

    colored_img_buf = BytesIO()
    colored_img.save(colored_img_buf, format="PNG")
    colored_img_buf.seek(0)

    return colored_img_buf
