"""
train_openmic_gpu.py
====================
State-of-the-art GPU-accelerated multi-label instrument tagging engine for OpenMIC-2018.

Architecture:
  - AudioResNet-SE: Residual CNN with Squeeze-and-Excitation (SE) Channel Attention
  - Dual Global Pooling: GlobalAveragePooling2D + GlobalMaxPooling2D
  - SpecAugment: Frequency Masking & Time Masking
  - Loss: BCEWithLogitsLoss with dynamically computed pos_weight to eliminate class imbalance
  - Acceleration: PyTorch 2.5.1 + CUDA 12.1 + FP16 Automatic Mixed Precision (AMP)
  - Target GPU: NVIDIA GeForce RTX 3050 6GB

Output: backend/app/models/audiotag_model_v1.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Setup Paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR    = os.path.join(SCRIPT_DIR, "openmic-2018")
AUDIO_DIR   = os.path.join(DATA_DIR, "audio")
LABELS_JSON = os.path.join(DATA_DIR, "clean_multilabels.json")
CACHE_X     = os.path.join(DATA_DIR, "cache_X.npy")
CACHE_Y     = os.path.join(DATA_DIR, "cache_y.npy")
MODEL_OUT   = os.path.join(PROJECT_ROOT, "backend", "app", "models", "audiotag_model_v1.pt")

INSTRUMENTS = [
    'accordion', 'bass', 'cello', 'clarinet', 'cymbals', 'drums', 'flute',
    'guitar', 'mallet_percussion', 'mandolin', 'piano', 'saxophone',
    'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

NUM_CLASSES   = len(INSTRUMENTS)   # 18
SR            = 22050
DURATION      = 10.0               # 10 seconds
N_SAMPLES     = int(SR * DURATION) # 220,500
N_MELS        = 128
IMG_W         = 128
FMAX          = 8000
DEFAULT_LR    = 1e-3
DEFAULT_BATCH = 32
DEFAULT_EPOCHS= 30
RANDOM_SEED   = 42


# ==============================================================================
# AUDIO PREPROCESSING
# ==============================================================================
def audio_to_melspec(file_path):
    """Load audio and convert to 128x128 Log-Mel spectrogram normalized to [0, 1]."""
    try:
        y, _ = librosa.load(file_path, sr=SR, mono=True)
    except Exception:
        return None

    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=IMG_W, axis=1)

    min_val = mel_db.min()
    max_val = mel_db.max()
    mel_norm = (mel_db - min_val) / (max_val - min_val + 1e-8)
    return mel_norm.astype(np.float32)


# ==============================================================================
# DATASET & SPECAUGMENT
# ==============================================================================
class OpenMICDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X  # (N, 1, 128, 128) numpy float32
        self.y = y  # (N, 18) numpy float32
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.y[idx]).float()

        if self.augment:
            # SpecAugment: Frequency Masking
            if torch.rand(1).item() > 0.5:
                f_band = int(torch.randint(4, 16, (1,)).item())
                f_start = int(torch.randint(0, N_MELS - f_band, (1,)).item())
                x[:, f_start:f_start + f_band, :] = 0.0

            # SpecAugment: Time Masking
            if torch.rand(1).item() > 0.5:
                t_band = int(torch.randint(4, 16, (1,)).item())
                t_start = int(torch.randint(0, IMG_W - t_band, (1,)).item())
                x[:, :, t_start:t_start + t_band] = 0.0

        return x, y


def _process_track_item(item):
    track_id, label_dict = item
    prefix = track_id[:3]
    ogg_path = os.path.join(AUDIO_DIR, prefix, f"{track_id}.ogg")
    if not os.path.exists(ogg_path):
        return None
    spec = audio_to_melspec(ogg_path)
    if spec is None:
        return None
    label_vec = np.array([label_dict.get(inst, 0) for inst in INSTRUMENTS], dtype=np.float32)
    return spec, label_vec


def load_data(max_samples=None, use_cache=True):
    """Load or build the OpenMIC dataset with disk caching."""
    if use_cache and os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        print("[INFO] Loading cached dataset from disk...")
        X = np.load(CACHE_X)
        y = np.load(CACHE_Y)
        if max_samples and len(X) > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
        print(f"   Loaded from cache: X={X.shape}  y={y.shape}")
        if X.ndim == 3:
            X = X[:, np.newaxis, ...]
        elif X.ndim == 4 and X.shape[-1] == 1:
            X = np.transpose(X, (0, 3, 1, 2))  # (N, 1, 128, 128)
        return X, y

    print("[INFO] Loading clean labels mapping...")
    with open(LABELS_JSON, "r") as f:
        labels_dict = json.load(f)

    items = list(labels_dict.items())
    if max_samples:
        items = items[:max_samples]

    print(f"[INFO] Extracting spectrograms across {os.cpu_count() or 8} CPU threads...")
    X, y = [], []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        results = list(tqdm(ex.map(_process_track_item, items), total=len(items), desc="Extracting"))

    for res in results:
        if res is not None:
            spec, label_vec = res
            X.append(spec)
            y.append(label_vec)

    X = np.array(X)[:, np.newaxis, ...]  # (N, 1, 128, 128)
    y = np.array(y)                     # (N, 18)

    if not max_samples and len(X) > 0:
        print("[INFO] Saving cache to disk...")
        # Save as (N, 128, 128, 1) for compatibility
        np.save(CACHE_X, np.transpose(X, (0, 2, 3, 1)))
        np.save(CACHE_Y, y)

    return X, y


# ==============================================================================
# SQUEEZE-AND-EXCITATION ATTENTION + RESIDUAL BLOCKS
# ==============================================================================
class SqueezeExcitation(nn.Module):
    """SE Attention block to model inter-channel instrument dependencies."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class SEBasicBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation channel attention."""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SqueezeExcitation(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


# ==============================================================================
# SOTA AUDIORESNET-SE ARCHITECTURE
# ==============================================================================
class AudioResNetSE(nn.Module):
    """
    Modern Audio ResNet with SE Attention & Dual (GAP+GMP) Pooling.
    Output: 18 unnormalized logits for BCEWithLogitsLoss.
    """
    def __init__(self, num_classes=18):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2)  # (32, 64, 64)
        )

        # Stage 1: 32 -> 64
        self.layer1 = nn.Sequential(
            SEBasicBlock(32, 64, stride=2),   # (64, 32, 32)
            SEBasicBlock(64, 64, stride=1)
        )

        # Stage 2: 64 -> 128
        self.layer2 = nn.Sequential(
            SEBasicBlock(64, 128, stride=2),  # (128, 16, 16)
            SEBasicBlock(128, 128, stride=1)
        )

        # Stage 3: 128 -> 256
        self.layer3 = nn.Sequential(
            SEBasicBlock(128, 256, stride=2), # (256, 8, 8)
            SEBasicBlock(256, 256, stride=1)
        )

        # Dual Pooling (Average + Max) -> 256 * 2 = 512 dims
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Dual Pooling
        gap = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        gmp = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        feat = torch.cat([gap, gmp], dim=1)  # (Batch, 512)

        logits = self.head(feat)
        return logits


# ==============================================================================
# MAIN TRAINING ROUTINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="GPU Training AudioTag AI on RTX 3050")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 65)
    print(f"  AudioTag AI — GPU AudioResNet-SE Training Engine")
    print(f"  Target Device: {device} ({gpu_name})")
    print("=" * 65)

    # 1. Load Data
    X, y = load_data(max_samples=args.max_samples, use_cache=not args.no_cache)
    print(f"\n[INFO] Dataset shape: X={X.shape}  y={y.shape}")

    # 2. Compute class imbalance pos_weights
    pos_counts = y.sum(axis=0)
    total_samples = len(y)
    pos_weights = (total_samples - pos_counts) / (pos_counts + 1e-5)
    pos_weights = np.clip(pos_weights, 1.0, 25.0)  # Bound extreme weights
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    print("\n[INFO] Class Distribution & Imbalance Weights (pos_weight):")
    for i, inst in enumerate(INSTRUMENTS):
        pct = (pos_counts[i] / total_samples) * 100
        print(f"   {inst:<20} {pct:5.1f}% present  |  pos_weight: {pos_weights[i]:.2f}")

    # 3. Train / Val / Test Splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15 / 0.90, random_state=RANDOM_SEED
    )
    print(f"\n[INFO] Split -- Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_ds = OpenMICDataset(X_train, y_train, augment=True)
    val_ds   = OpenMICDataset(X_val, y_val, augment=False)
    test_ds  = OpenMICDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # 4. Initialize Model, Optimizer, Loss, Scaler
    model = AudioResNetSE(num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_auc = 0.0
    patience = 7
    patience_counter = 0
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_OUT)), exist_ok=True)

    print(f"\n[INFO] Commencing GPU Training for up to {args.epochs} epochs (AMP enabled)...")

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                logits = model(x_b)
                loss = criterion(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(x_b)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                    logits = model(x_b)
                    loss = criterion(logits, y_b)

                val_loss += loss.item() * len(x_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y_b.cpu().numpy())

        val_loss /= len(val_ds)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        # Macro AUROC
        try:
            val_auc = roc_auc_score(val_targets, val_preds, average="macro")
        except Exception:
            val_auc = 0.50

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'instruments': INSTRUMENTS
            }, MODEL_OUT)
            print(f"   --> Best checkpoint saved to {os.path.basename(MODEL_OUT)} (AUROC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # 6. Evaluation on Held-Out Test Set
    print("\n" + "=" * 65)
    print("  Evaluating Best Model on Held-out Test Set")
    print("=" * 65)

    checkpoint = torch.load(MODEL_OUT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(y_b.numpy())

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_auc = roc_auc_score(test_targets, test_preds, average="macro")
    print(f"\nOverall Test Macro AUROC: {test_auc:.4f}")

    # Per-class metrics table
    test_preds_bin = (test_preds >= 0.50).astype(int)
    print("\nPer-Instrument Breakdown (Threshold = 0.50):")
    print(f"   {'Instrument':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUROC':<10}")
    print("   " + "-" * 60)

    for i, inst in enumerate(INSTRUMENTS):
        tp = ((test_preds_bin[:, i] == 1) & (test_targets[:, i] == 1)).sum()
        fp = ((test_preds_bin[:, i] == 1) & (test_targets[:, i] == 0)).sum()
        fn = ((test_preds_bin[:, i] == 0) & (test_targets[:, i] == 1)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        try:
            cls_auc = roc_auc_score(test_targets[:, i], test_preds[:, i])
        except Exception:
            cls_auc = 0.50
        print(f"   {inst:<20} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f} {cls_auc:<10.3f}")

    print("\n[SUCCESS] PyTorch AudioResNet-SE training and evaluation completed!")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
