import os
import zipfile
import logging
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

def build_dataset_from_zip_stream(zip_stream) -> pd.DataFrame:
    """ZIP 파일 스트림에서 이미지와 메타데이터를 읽어 DataFrame 생성"""
    data_list = []

    try:
        zf = zipfile.ZipFile(zip_stream, 'r')
        for file_path in zf.namelist():
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')) and not file_path.startswith('__MACOSX'):
                try:
                    filename = os.path.basename(file_path)
                    if not filename: continue

                    filename_no_ext = filename.rsplit('.', 1)[0]
                    parts = filename_no_ext.split('_')

                    if len(parts) == 2:
                        lot_name, wafer_num = parts

                        with zf.open(file_path) as image_file:
                            image_stream = BytesIO(image_file.read())
                            image = Image.open(image_stream)
                            image_array = np.array(image)

                            data_list.append({
                                'lotName': lot_name,
                                'waferIndex': int(wafer_num),
                                'waferMap': image_array
                            })
                    else:
                        logger.warning(f"파일명 형식이 맞지 않습니다 - {filename}")
                except Exception as e:
                    logger.error(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
    finally:
        if 'zf' in locals() and zf:
            zf.close()
            
    return pd.DataFrame(data_list)