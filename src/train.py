"""
train.py
--------
Training loop for the multi-branch ResumeMatchNet.

Handles:
  - Train / val / test split (60/20/20, seed=42)
  - Adam optimiser with cosine learning-rate annealing
  - Early stopping (patience=10 epochs)
  - Checkpointing best model (lowest val RMSE)
  - Learning-curve logging + matplotlib plots
  - Accuracy and weighted F1 (threshold=0.5) alongside RMSE/MAE/R²

Usage:
    python train.py [--epochs 50] [--batch_size 64] [--lr 1e-3] [--hidden_dim 512]
                    [--dropout 0.3] [--refit_embeddings]
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score

from dataset import ResumeDataset, split_dataset
from model import ResumeMatchNet

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Threshold to convert regression output → binary class for Accuracy / F1
# (matched_score ≥ 0.5 → "good match",  < 0.5 → "poor match")
MATCH_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for b1, b2, y in loader:
            b1, b2, y = b1.to(device), b2.to(device), y.to(device)
            pred = model(b1, b2)
            total_loss += criterion(pred, y).item() * len(y)
            preds_all.append(pred.cpu().numpy())
            targets_all.append(y.cpu().numpy())
    preds_all   = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    mse  = total_loss / len(targets_all)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(preds_all - targets_all))
    ss_res = np.sum((targets_all - preds_all) ** 2)
    ss_tot = np.sum((targets_all - targets_all.mean()) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-12)
    # Binary classification metrics at MATCH_THRESHOLD
    pred_bin   = (preds_all   >= MATCH_THRESHOLD).astype(int)
    target_bin = (targets_all >= MATCH_THRESHOLD).astype(int)
    acc = accuracy_score(target_bin, pred_bin)
    f1  = f1_score(target_bin, pred_bin, average="weighted", zero_division=0)
    return rmse, mae, r2, acc, f1, preds_all, targets_all


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    csv_path: str,
    cache_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    patience: int = 10,
    refit_embeddings: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = ResumeDataset(csv_path, cache_dir=cache_dir, fit=refit_embeddings)
    train_set, val_set, test_set = split_dataset(dataset, train=0.60, val=0.20, test=0.20,
                                                  seed=RANDOM_SEED)
    print(f"Split: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ResumeMatchNet(
        branch1_dim=dataset.branch1_dim,
        branch2_dim=dataset.branch2_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    criterion  = nn.MSELoss()
    optimiser  = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_rmse = float("inf")
    patience_counter = 0
    history = {"train_rmse": [], "val_rmse": [], "train_mae": [], "val_mae": [],
               "train_r2":   [], "val_r2":   [], "train_acc": [], "val_acc": [],
               "train_f1":   [], "val_f1":   [], "lr": []}
    ckpt_path = os.path.join(output_dir, "best_model.pt")

    print(f"\nTraining for up to {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for b1, b2, y in train_loader:
            b1, b2, y = b1.to(device), b2.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(b1, b2)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(y)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_rmse_e, train_mae_e, train_r2_e, train_acc_e, train_f1_e, _, _ = \
            _evaluate(model, train_loader, device, criterion)
        val_rmse_e, val_mae_e, val_r2_e, val_acc_e, val_f1_e, _, _ = \
            _evaluate(model, val_loader, device, criterion)

        history["train_rmse"].append(train_rmse_e)
        history["val_rmse"].append(val_rmse_e)
        history["train_mae"].append(train_mae_e)
        history["val_mae"].append(val_mae_e)
        history["train_r2"].append(train_r2_e)
        history["val_r2"].append(val_r2_e)
        history["train_acc"].append(train_acc_e)
        history["val_acc"].append(val_acc_e)
        history["train_f1"].append(train_f1_e)
        history["val_f1"].append(val_f1_e)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_RMSE={train_rmse_e:.4f}  val_RMSE={val_rmse_e:.4f}  "
              f"val_R²={val_r2_e:.4f}  val_Acc={val_acc_e:.4f}  val_F1={val_f1_e:.4f}  "
              f"lr={current_lr:.2e}  [{elapsed:.1f}s]")

        # Checkpoint + early stopping
        if val_rmse_e < best_val_rmse:
            best_val_rmse = val_rmse_e
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_rmse": val_rmse_e}, ckpt_path)
            print(f"  ✓ New best val RMSE={best_val_rmse:.4f} — saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break

    # ── Final evaluation on test set ──────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    test_rmse, test_mae, test_r2, test_acc, test_f1, preds, targets = \
        _evaluate(model, test_loader, device, criterion)
    print(f"\nFinal Test Results  (classification threshold = {MATCH_THRESHOLD}):")
    print(f"  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  R²={test_r2:.4f}"
          f"  Acc={test_acc:.4f}  F1(wtd)={test_f1:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    epochs_run = len(history["train_rmse"])
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Learning curve — RMSE
    ax = axes[0]
    ax.plot(range(1, epochs_run + 1), history["train_rmse"], label="Train RMSE", color="steelblue")
    ax.plot(range(1, epochs_run + 1), history["val_rmse"],   label="Val RMSE",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning Curve — RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning curve — R²
    ax = axes[1]
    ax.plot(range(1, epochs_run + 1), history["train_r2"], label="Train R²", color="steelblue")
    ax.plot(range(1, epochs_run + 1), history["val_r2"],   label="Val R²",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Learning Curve — R²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning curve — Accuracy & F1
    ax = axes[2]
    ax.plot(range(1, epochs_run + 1), history["train_acc"], label="Train Acc", color="steelblue")
    ax.plot(range(1, epochs_run + 1), history["val_acc"],   label="Val Acc",   color="tomato")
    ax.plot(range(1, epochs_run + 1), history["train_f1"],  label="Train F1",  color="steelblue", linestyle="--")
    ax.plot(range(1, epochs_run + 1), history["val_f1"],    label="Val F1",    color="tomato",    linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Accuracy & F1  (threshold={MATCH_THRESHOLD})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Test scatter
    ax = axes[3]
    ax.scatter(targets, preds, alpha=0.2, s=10, color="teal")
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True matched_score")
    ax.set_ylabel("Predicted matched_score")
    ax.set_title(f"Test predictions  (RMSE={test_rmse:.4f}, R²={test_r2:.4f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("ResumeMatchNet — Primary Model Results", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "primary_model_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_path}")

    # Save history
    np.save(os.path.join(output_dir, "training_history.npy"), history)

    return {
        "test_rmse": test_rmse,
        "test_mae":  test_mae,
        "test_r2":   test_r2,
        "test_acc":  test_acc,
        "test_f1":   test_f1,
        "history":   history,
        "ckpt_path": ckpt_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResumeMatchNet")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden_dim",   type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--refit_embeddings", action="store_true",
                        help="Recompute SBERT and Word2Vec embeddings")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train(
        csv_path=os.path.join(base, "data", "cleaned_resume_data.csv"),
        cache_dir=os.path.join(base, "data", "cache"),
        output_dir=os.path.join(base, "data", "primary_model"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        refit_embeddings=args.refit_embeddings,
    )
