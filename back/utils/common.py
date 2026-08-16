import random
import numpy as np
import torch
from config.settings import SEED

def set_seed(seed=SEED):
    """재현성 확보를 위한 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    """DataLoader 워커의 시드를 설정하는 함수"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 불량률 높은 순서대로 lot 리스트 반환
def get_lot_defect_ranking_with_error_counts(df_image):
    """불량률 높은 순서대로 lot 리스트 반환 (불량률 = 불량 / (불량 + 정상))"""
    lot_stats = {}

    for _, row in df_image.iterrows():
        lot = row['lotName']
        pred = row['pred_value']

        if lot not in lot_stats:
            lot_stats[lot] = {
                'defect_count': 0,
                'normal_count': 0,
                'error_detail': {i: 0 for i in range(9)}
            }

        if 0 <= pred <= 7:
            lot_stats[lot]['defect_count'] += 1
        elif pred == 8:
            lot_stats[lot]['normal_count'] += 1
        lot_stats[lot]['error_detail'][pred] += 1  # 정상도 포함

    # 불량률 계산 및 정렬용 dict 구성
    lot_defect_info = {}
    for lot, stats in lot_stats.items():
        total = stats['defect_count'] + stats['normal_count']
        defect_rate = stats['defect_count'] / total if total > 0 else 0

        lot_defect_info[lot] = {
            'error_detail': stats['error_detail'],
            'defect_rate': defect_rate
        }

    # 불량률 기준 정렬
    sorted_lot_defect_info = dict(
        sorted(lot_defect_info.items(), key=lambda x: x[1]['defect_rate'], reverse=True)
    )

    result = {
        lot: detail['error_detail']
        for lot, detail in sorted_lot_defect_info.items()
    }

    return result