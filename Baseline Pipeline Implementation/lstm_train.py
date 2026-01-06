import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from model_baseline_B import LSTMSpectrogramClassifier

# ==============================================================
# 1. Load and prepare data
# ==============================================================
print(" Loading preprocessed data...")
X = np.load("results/X_train.npy")
y = np.load("results/y_train.npy").astype(int) - 1  # shift labels [1–5] → [0–4]

# Normalize globally
X = (X - X.mean()) / (X.std() + 1e-8)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split into train/validation
train_size = int(0.8 * len(X))
val_size = len(X) - train_size
train_data, val_data = random_split(TensorDataset(X, y), [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f" Train size: {len(train_data)}, Validation size: {len(val_data)}")

# ==============================================================
# 2. Initialize model, optimizer, scheduler
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

model = LSTMSpectrogramClassifier(
    n_mels=X.shape[2],
    hidden_size=256,
    num_layers=3,
    dropout=0.4,
    num_classes=5
)  # automatically moved to device

# Weighted loss for imbalance
class_counts = torch.bincount(y)
weights = (1.0 / class_counts.float()) * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)
# AMP (mixed precision)
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler(enabled=(device.type == "cuda"))

torch.backends.cudnn.benchmark = True

# ==============================================================
# 3. Training setup
# ==============================================================
epochs = 50
best_val_f1 = 0
patience, patience_counter = 8, 0
metrics = []

print("\n Starting fine-tuned LSTM training...\n")

for epoch in range(1, epochs + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

    train_acc = correct / total
    train_loss /= len(train_loader)

    # ------------------ Validation ------------------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_val).sum().item()
            val_total += y_val.size(0)
            y_true.extend(y_val.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_f1 = f1_score(y_true, y_pred, average="macro")

    # Step the LR scheduler
    scheduler.step(val_loss)

    metrics.append([epoch, train_loss, train_acc, val_loss, val_acc, val_f1])
    print(f"Epoch [{epoch}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f} || "
          f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "results/lstm_baseline_B_best.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print(f" Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs)")
        break

# ==============================================================
# 4. Save metrics & plots
# ==============================================================
os.makedirs("results", exist_ok=True)
metrics_df = pd.DataFrame(metrics, columns=["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "ValF1"])
metrics_df.to_csv("results/metrics_lstm_finetuned.csv", index=False)

print("\n Fine-tuning complete.")
print(f" Best Validation F1: {best_val_f1:.3f}")
print(" Metrics saved to results/metrics_lstm_finetuned.csv")

# Plot results
plt.figure(figsize=(8,5))
plt.plot(metrics_df["Epoch"], metrics_df["ValAcc"], label="Val Accuracy", marker="o")
plt.plot(metrics_df["Epoch"], metrics_df["ValF1"], label="Val F1", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Fine-Tuned LSTM Validation Performance")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lstm_finetuned_validation_curves.png")
plt.close()

print(" Validation curves saved to plots/lstm_finetuned_validation_curves.png")

# Save final model as well
torch.save(model.state_dict(), "results/lstm_baseline_B_final.pth")
print(" Final model saved to results/lstm_baseline_B_final.pth")
