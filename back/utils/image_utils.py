import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from config.settings import TARGET_SIZE

def resize_and_pad(image, target_size=TARGET_SIZE):
    """이미지의 가로세로 비율을 유지하면서 리사이즈하고 패딩"""
    target_h, target_w = target_size
    h, w = image.shape
    
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    padded_image = np.zeros(target_size, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_image[top:top+new_h, left:left+new_w] = resized_image
    
    return padded_image

def convert_into_colored_img(image_bytes_io):
    """이미지를 컬러 이미지로 변환"""
    image = Image.open(image_bytes_io)
    image = np.array(image)
    h, w = image.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    rgba[image == 0] = (0, 0, 0, 0)
    rgba[image == 1] = (192, 192, 192, 255)
    rgba[image == 2] = (255, 0, 0, 255)
    
    colored_img = Image.fromarray(rgba, mode="RGBA")
    
    colored_img_buf = BytesIO()
    colored_img.save(colored_img_buf, format="PNG")
    colored_img_buf.seek(0)
    
    return colored_img_buf