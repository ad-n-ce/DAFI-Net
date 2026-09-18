import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from datasets.fastmri_dataset import FastMRIDataset
from models.dafi_net import DAFINet


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = r"path"

MODEL_PATH = (
    r"path"
)

OUTPUT_DIR = (
    r"path"
)

BATCH_SIZE = 1

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 60)
print("DAFI-Net Evaluation")
print("=" * 60)
print("Device:", DEVICE)
print("=" * 60)


# =====================================================
# DATASET
# =====================================================

test_dataset = FastMRIDataset(
    root_dir=DATA_DIR,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(
    f"Test Samples: {len(test_dataset)}"
)

print("=" * 60)


# =====================================================
# MODEL
# =====================================================

model = DAFINet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Loaded Successfully")
print("=" * 60)


# =====================================================
# EVALUATION
# =====================================================

psnr_scores = []
ssim_scores = []
mae_scores = []

with torch.no_grad():

    for idx, batch in enumerate(
        tqdm(test_loader)
    ):

        kspace = batch["kspace"].to(
            DEVICE,
            non_blocking=True
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        pred = model(
            kspace
        )

        pred_np = (
            pred.squeeze()
            .cpu()
            .numpy()
        )

        target_np = (
            target.squeeze()
            .cpu()
            .numpy()
        )

        pred_np = np.clip(
            pred_np,
            0,
            1
        )

        psnr = peak_signal_noise_ratio(
            target_np,
            pred_np,
            data_range=1.0
        )

        ssim = structural_similarity(
            target_np,
            pred_np,
            data_range=1.0
        )

        mae = np.mean(
            np.abs(
                pred_np -
                target_np
            )
        )

        psnr_scores.append(
            psnr
        )

        ssim_scores.append(
            ssim
        )

        mae_scores.append(
            mae
        )

        # --------------------------------
        # Save reconstruction
        # --------------------------------

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{idx:06d}.npy"
            ),
            pred_np
        )

# =====================================================
# RESULTS
# =====================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Mean PSNR : "
    f"{np.mean(psnr_scores):.4f} dB"
)

print(
    f"Std PSNR  : "
    f"{np.std(psnr_scores):.4f}"
)

print()

print(
    f"Mean SSIM : "
    f"{np.mean(ssim_scores):.4f}"
)

print(
    f"Std SSIM  : "
    f"{np.std(ssim_scores):.4f}"
)

print()

print(
    f"Mean MAE  : "
    f"{np.mean(mae_scores):.6f}"
)

print("=" * 60)