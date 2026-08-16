import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from config.settings import TARGET_SIZE
from utils.image_utils import resize_and_pad

class WaferDataset(Dataset):
    def __init__(self, images, labels, lot_names, transforms=None):
        self.images = images
        self.labels = labels
        self.lot_names = lot_names
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = int(self.labels[idx])
        lot = self.lot_names[idx]

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.uint8)

        target_size = TARGET_SIZE[0]
        h, w = x.shape
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        padded_x = np.zeros((target_size, target_size), dtype=np.uint8)
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        padded_x[top:top+new_h, left:left+new_w] = resized_x
        x = padded_x

        x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0) / 2.0

        if self.transforms:
            x = self.transforms(x)

        return x, y, lot

class WaferInferenceDataset(Dataset):
    def __init__(self, wafer_series, target_size=TARGET_SIZE):
        self.wafer_series = wafer_series.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.wafer_series)

    def __getitem__(self, idx):
        x = self.wafer_series.iloc[idx]
        
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.uint8)
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
            
        x = resize_and_pad(x, self.target_size)
        xt = torch.from_numpy(x).float().unsqueeze(0) / 2.0
        
        return xt