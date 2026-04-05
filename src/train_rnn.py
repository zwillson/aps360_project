"""
train_rnn.py
------------
Training loop for the RNN variant of ResumeMatchNet.

Same training recipe as train.py:
  - 60/20/20 split (seed=42) — identical indices to SBERT model
  - Adam + CosineAnnealingLR
  - Early stopping (patience=10)
  - RMSE / MAE / R² / Accuracy / F1 metrics

Usage:
    python train_rnn.py [--epochs 50] [--batch_size 64] [--lr 1e-3] [--refit]
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from rnn_dataset import RNNResumeDataset, rnn_collate_fn
from rnn_model import ResumeMatchNetRNN

RANDOM_SEED = 42
MATCH_THRESHOLD = 0.5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_indices(n, train_frac=0.60, val_frac=0.20, seed=42):
    """Return train/val/test index arrays — same split as train.py."""
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _to_device(branch1, w2v, tc, tr, te, y, device):
    """Move a batch to the target device."""
    return (
        branch1.to(device), w2v.to(device),
        (tc[0].to(device), tc[1].to(device)),
        (tr[0].to(device), tr[1].to(device)),
        (te[0].to(device), te[1].to(device)),
        y.to(device),
    )


def _evaluate(model, loader, device, criterion):
    """Compute metrics on a DataLoader (same signature as train.py)."""
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for branch1, w2v, tc, tr, te, y in loader:
            branch1, w2v, tc, tr, te, y = _to_device(branch1, w2v, tc, tr, te, y, device)
            pred = model(branch1, w2v, tc, tr, te)
            total_loss += criterion(pred, y).item() * len(y)
            preds_all.append(pred.cpu().numpy())
            targets_all.append(y.cpu().numpy())
    preds   = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    mse  = total_loss / len(targets)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    pred_bin   = (preds   >= MATCH_THRESHOLD).astype(int)
    target_bin = (targets >= MATCH_THRESHOLD).astype(int)
    acc = accuracy_score(target_bin, pred_bin)
    f1  = f1_score(target_bin, pred_bin, average="weighted", zero_division=0)
    return rmse, mae, r2, acc, f1, preds, targets


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
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
    dataset = RNNResumeDataset(csv_path, cache_dir=cache_dir, fit=refit_embeddings)

    train_idx, val_idx, test_idx = _split_indices(len(dataset), seed=RANDOM_SEED)
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)
    print(f"Split: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=rnn_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              collate_fn=rnn_collate_fn, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              collate_fn=rnn_collate_fn, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    model = ResumeMatchNetRNN(
        branch1_dim=dataset.branch1_dim,
        vocab_size=len(dataset.vocab),
        w2v_dim=dataset.w2v.shape[1],
        embed_dim=128,
        rnn_hidden=192,
        rnn_layers=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_rmse = float("inf")
    patience_counter = 0
    history = {
        "train_rmse": [], "val_rmse": [],
        "train_mae":  [], "val_mae":  [],
        "train_r2":   [], "val_r2":   [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
        "lr": [],
    }
    ckpt_path = os.path.join(output_dir, "best_rnn_model.pt")

    print(f"\nTraining for up to {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for branch1, w2v, tc, tr, te, y in train_loader:
            branch1, w2v, tc, tr, te, y = _to_device(branch1, w2v, tc, tr, te, y, device)
            optimiser.zero_grad()
            pred = model(branch1, w2v, tc, tr, te)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        tr_rmse, tr_mae, tr_r2, tr_acc, tr_f1, _, _ = \
            _evaluate(model, train_loader, device, criterion)
        vl_rmse, vl_mae, vl_r2, vl_acc, vl_f1, _, _ = \
            _evaluate(model, val_loader, device, criterion)

        for k, v in [
            ("train_rmse", tr_rmse), ("val_rmse", vl_rmse),
            ("train_mae",  tr_mae),  ("val_mae",  vl_mae),
            ("train_r2",   tr_r2),   ("val_r2",   vl_r2),
            ("train_acc",  tr_acc),  ("val_acc",  vl_acc),
            ("train_f1",   tr_f1),   ("val_f1",   vl_f1),
            ("lr", current_lr),
        ]:
            history[k].append(v)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_RMSE={tr_rmse:.4f}  val_RMSE={vl_rmse:.4f}  "
              f"val_R\u00b2={vl_r2:.4f}  val_Acc={vl_acc:.4f}  val_F1={vl_f1:.4f}  "
              f"lr={current_lr:.2e}  [{elapsed:.1f}s]")

        if vl_rmse < best_val_rmse:
            best_val_rmse = vl_rmse
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_rmse": vl_rmse,
                "vocab_size": len(dataset.vocab),
                "branch1_dim": dataset.branch1_dim,
            }, ckpt_path)
            print(f"  \u2713 New best val RMSE={best_val_rmse:.4f} \u2014 saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {patience} epochs).")
                break

    # ── Final test evaluation ────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    test_rmse, test_mae, test_r2, test_acc, test_f1, preds, targets = \
        _evaluate(model, test_loader, device, criterion)
    print(f"\nFinal Test Results  (threshold={MATCH_THRESHOLD}):")
    print(f"  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  R\u00b2={test_r2:.4f}"
          f"  Acc={test_acc:.4f}  F1(wtd)={test_f1:.4f}")

    # ── Plots ────────────────────────────────────────────────────────────
    epochs_run = len(history["train_rmse"])
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    ax = axes[0]
    ax.plot(range(1, epochs_run+1), history["train_rmse"], label="Train", color="steelblue")
    ax.plot(range(1, epochs_run+1), history["val_rmse"],   label="Val",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_title("RMSE"); ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, epochs_run+1), history["train_r2"], label="Train", color="steelblue")
    ax.plot(range(1, epochs_run+1), history["val_r2"],   label="Val",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("R\u00b2"); ax.set_title("R\u00b2"); ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(range(1, epochs_run+1), history["train_acc"], label="Train Acc", color="steelblue")
    ax.plot(range(1, epochs_run+1), history["val_acc"],   label="Val Acc",   color="tomato")
    ax.plot(range(1, epochs_run+1), history["train_f1"],  label="Train F1",  color="steelblue", ls="--")
    ax.plot(range(1, epochs_run+1), history["val_f1"],    label="Val F1",    color="tomato",    ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Accuracy & F1")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.scatter(targets, preds, alpha=0.2, s=10, color="teal")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title(f"Test (RMSE={test_rmse:.4f}, R\u00b2={test_r2:.4f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("ResumeMatchNet-RNN \u2014 Results", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rnn_model_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    np.save(os.path.join(output_dir, "rnn_training_history.npy"), history)

    return {
        "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2,
        "test_acc": test_acc, "test_f1": test_f1, "history": history,
    }


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResumeMatchNet-RNN")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int,   default=512)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--refit",      action="store_true",
                        help="Rebuild RNN vocabulary and Word2Vec from scratch")
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
        refit_embeddings=args.refit,
    )
