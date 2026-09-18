import torch
from torch.utils.data import DataLoader
from datasets.fastmri_dataset import FastMRIDataset
from models.dafi_net import DAFINet
from losses.losses import DAFILoss
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = r"Path"

BATCH_SIZE = 2

EPOCHS = 25

LR = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
torch.backends.cudnn.benchmark = True

print("=" * 60)
print("DAFI-Net Training")
print("=" * 60)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 60)


# =====================================================
# DATASETS
# =====================================================

print("Loading datasets...")

train_dataset = FastMRIDataset(
    root_dir=DATA_DIR,
    split="train"
)

val_dataset = FastMRIDataset(
    root_dir=DATA_DIR,
    split="val"
)

print(
    f"Training Samples: {len(train_dataset)}"
)

print(
    f"Validation Samples: {len(val_dataset)}"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Datasets loaded successfully.")
print("=" * 60)

sample = train_dataset[0]

print("\nSample Verification")

print(
    "K-space Shape:",
    sample["kspace"].shape
)

print(
    "Target Shape:",
    sample["target"].shape
)

# =====================================================
# MODEL
# =====================================================

print("Building DAFI-Net...")

model = DAFINet().to(
    DEVICE
)

criterion = DAFILoss(
    lambda_ssim=0.10,
    # lambda_freq=0.05,
    lambda_edge=0.03,
    lambda_wavelet=0.10
).to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    verbose=True
)
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total Parameters: {total_params:,}"
)

print(
    f"Trainable Parameters: {trainable_params:,}"
)

print("=" * 60)


# =====================================================
# BEST MODEL DIRECTORY
# =====================================================

import os

BEST_MODEL_DIR = r"path"

os.makedirs(
    BEST_MODEL_DIR,
    exist_ok=True
)

best_val_loss = float("inf")

best_epoch = 0


# =====================================================
# TRAINING
# =====================================================

for epoch in range(EPOCHS):

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    print(
        f"\nStarting Epoch {epoch+1}/{EPOCHS}"
    )

    # =================================================
    # TRAINING
    # =================================================

    model.train()

    train_loss = 0.0

    train_bar = tqdm(
    train_loader,
    desc=f"Epoch [{epoch+1}/{EPOCHS}] Train",
    leave=False
    )

    for batch in train_bar:
        kspace = batch["kspace"].to(
            DEVICE,
            non_blocking=True
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        pred = model(
            kspace
        )

        loss, _ = criterion(
            pred,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{train_loss/(train_bar.n+1):.4f}"
        )

    train_loss /= len(train_loader)

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch [{epoch+1}/{EPOCHS}] Val",
            leave=False
        )

        for batch in val_bar:

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

            loss, _ = criterion(
                pred,
                target
            )

            val_loss += loss.item()
            val_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    # =================================================
    # SAVE BEST MODEL ONLY
    # =================================================

    is_best = False

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch + 1

        is_best = True

        best_model_path = os.path.join(
            BEST_MODEL_DIR,
            "DAFI-Net_best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss
            },
            best_model_path
        )

    # =================================================
    # REPORT
    # =================================================

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch+1}/{EPOCHS} Completed"
    )

    print(
        f"Training Loss   : {train_loss:.6f}"
    )

    print(
        f"Validation Loss : {val_loss:.6f}"
    )

    print(
        f"Best Val Loss   : {best_val_loss:.6f}"
    )

    print(
        f"Best Epoch      : {best_epoch}"
    )
    
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate  : {current_lr:.8f}"
    )
    if is_best:

        print(
            "\n NEW BEST MODEL SAVED"
        )

        print(
            f"Location: {best_model_path}"
        )

    if torch.cuda.is_available():

        memory_used = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Peak GPU Memory: "
            f"{memory_used:.2f} GB"
        )

    print("=" * 60)

print("\nTraining Completed Successfully.")

print(
    f"\nBest Model Epoch : {best_epoch}"
)

print(
    f"Best Validation Loss : "
    f"{best_val_loss:.6f}"
)
