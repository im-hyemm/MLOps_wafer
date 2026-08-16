import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import nullcontext
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from tqdm import tqdm

from core.model.architecture import SmallCNN
from core.data.dataset import WaferDataset
from core.evaluation.metrics import calculate_metrics
from core.evaluation.visualization import draw_cm_heatmap
from utils.common import set_seed, seed_worker
from config.settings import *
from config.paths import MODEL_DIR, NEW_MODEL_HEATMAP_PATH

class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = -np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_f1, model, optimizer, scheduler):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model, optimizer, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model, optimizer, scheduler)
            self.counter = 0

    def save_checkpoint(self, val_f1, model, optimizer, scheduler):
        if self.verbose:
            print(f'Validation F1 improved ({self.val_f1_max:.6f} --> {val_f1:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1

def process(dataset, isPresentation):
    """모델 학습 프로세스"""
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset['label_id'] = dataset['failureType'].map(LABEL2ID)

    X = dataset['waferMap'].values
    y = dataset['label_id'].values.astype(int)
    lots = dataset["lotName"].values
    
    X_train, X_temp, y_train, y_temp, lots_train, lots_temp = train_test_split(
        X, y, lots, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test, lots_val, lots_test = train_test_split(
        X_temp, y_temp, lots_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 오버샘플링
    X_train_df = pd.DataFrame({'waferMap': X_train, 'label_id': y_train, 'lotName': lots_train})
    average_size = int(X_train_df['label_id'].value_counts().mean())
    
    resampled_df_list = []
    for label in X_train_df['label_id'].unique():
        class_df = X_train_df[X_train_df['label_id'] == label]
        n_samples = average_size

        resampled_df = resample(
            class_df,
            replace=True,
            n_samples=n_samples,
            random_state=42
        )
        resampled_df_list.append(resampled_df)

    resampled_train_df = pd.concat(resampled_df_list)
        
    # 데이터셋 생성
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
    ])
    
    train_ds = WaferDataset(
        resampled_train_df['waferMap'].values,
        resampled_train_df['label_id'].values,
        resampled_train_df['lotName'].values,
        transforms=train_transforms
    )
    val_ds = WaferDataset(X_val, y_val, lots_val, transforms=None)
    test_ds = WaferDataset(X_test, y_test, lots_test, transforms=None)

    # DataLoader
    batch_size = PRESENTATION_BATCH_SIZE if isPresentation else BATCH_SIZE
    g = torch.Generator()
    g.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory_flag = True if device.type == "cuda" else False
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory_flag, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory_flag, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory_flag, worker_init_fn=seed_worker, generator=g)

    # 모델 저장 경로
    current_time = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    new_modelname = f"{current_time}_model.pth"
    model_location = os.path.join(MODEL_DIR, new_modelname)
    os.makedirs(os.path.dirname(model_location), exist_ok=True)
    
    # 학습 설정
    model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    epochs = PRESENTATION_EPOCHS if isPresentation else EPOCHS
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=SCHEDULER_PATIENCE)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext())
    
    def run_epoch(loader, train=True, test=False):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        all_preds, all_labels, all_lots = [], [], []

        for xb, yb, lot_name_batch in tqdm(loader, desc=f"Epoch {'Train' if train else ('Test' if test else 'Validation')}"):
            xb, yb = xb.to(device), yb.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with amp_ctx:
                        logits = model(xb)
                        loss = criterion(logits, yb)

            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(yb.detach().cpu().numpy())
            all_lots.extend(lot_name_batch)

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return loss_sum/total, correct/total, f1, all_labels, all_preds, all_lots

    # 학습 루프
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=model_location)

    for ep in range(1, epochs + 1):
        print(f"[Epoch {ep:03d}] starts...")
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(train_loader, train=True, test=False)
        val_loss, val_acc, val_f1, _, _, _ = run_epoch(val_loader, train=False, test=False)

        scheduler.step(val_f1)

        print(
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        early_stopping(val_f1, model, optimizer, scheduler)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    print(f"Best model saved with val macro-F1 = {early_stopping.best_score:.4f} → {model_location}\n")

    # 테스트 평가
    model = SmallCNN(num_classes=NUM_CLASSES)
    state = torch.load(model_location, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    _, test_acc, _, all_labels, all_preds, all_lots = run_epoch(test_loader, train=False, test=True)

    test_acc, test_macro_f1, test_macro_precision, test_macro_recall, metrics_by_cat = calculate_metrics(all_labels, all_preds)

    # Confusion Matrix
    _ = draw_cm_heatmap(all_labels, all_preds, NEW_MODEL_HEATMAP_PATH, palette='Peach', normalize='true')
    
    all_labels = np.array(all_labels).tolist()
    all_preds = np.array(all_preds).tolist()

    return test_acc, test_macro_f1, test_macro_precision, test_macro_recall, metrics_by_cat, all_labels, all_preds, all_lots, model_location