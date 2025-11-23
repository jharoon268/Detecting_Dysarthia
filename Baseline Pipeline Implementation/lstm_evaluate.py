import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from model_baseline_B import LSTMSpectrogramClassifier

# -------------------------------------------------------------
# 1. Load preprocessed data
# -------------------------------------------------------------
print("Loading preprocessed data...")

X = np.load("results/X_train.npy")
y = np.load("results/y_train.npy")
y = y - 1  # Shift labels [1–5] → [0–4]

num_classes = len(np.unique(y))
print(f" Data loaded: {X.shape}, {y.shape}, Classes: {num_classes}")

# train/val split
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# -------------------------------------------------------------
# 2. Prepare model and device
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMSpectrogramClassifier(n_mels=X.shape[2], num_classes=num_classes).to(device)
model_path = "results/lstm_baseline_B_final.pth"

print(f"Loading model from {model_path}")
state = torch.load(model_path, map_location=device)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Model loaded (non-strict).\nMissing: {missing}\nUnexpected: {unexpected}")

model.eval()

# -------------------------------------------------------------
# 3. Run inference on validation set
# -------------------------------------------------------------
print("\nRunning inference on validation set...")

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
batch_size = 32
probs, preds = [], []

with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
        batch = X_val_tensor[i:i + batch_size]
        outputs = model(batch)
        softmax_out = F.softmax(outputs, dim=1)
        probs.append(softmax_out.cpu().numpy())
        preds.extend(torch.argmax(softmax_out, dim=1).cpu().numpy())

probs = np.concatenate(probs, axis=0)
preds = np.array(preds)
print(f"Inference complete. Shape of probs: {probs.shape}")

# -------------------------------------------------------------
# 4. Compute metrics
# -------------------------------------------------------------
y_val_bin = label_binarize(y_val, classes=np.arange(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

macro_auc = roc_auc_score(y_val_bin, probs, average="macro")
micro_auc = roc_auc_score(y_val_bin, probs, average="micro")
acc = accuracy_score(y_val, preds)
f1_weighted = f1_score(y_val, preds, average="weighted")
precision_weighted = precision_score(y_val, preds, average="weighted")
recall_weighted = recall_score(y_val, preds, average="weighted")

# Per-class metrics
precision_per_class = precision_score(y_val, preds, average=None, labels=np.arange(num_classes))
recall_per_class = recall_score(y_val, preds, average=None, labels=np.arange(num_classes))
f1_per_class = f1_score(y_val, preds, average=None, labels=np.arange(num_classes))

print(f"\nValidation Results:")
print(f" Accuracy: {acc:.3f} | Weighted F1: {f1_weighted:.3f} | Macro AUC: {macro_auc:.3f}")

# -------------------------------------------------------------
# 5. Plot ROC–AUC Curves
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC={roc_auc[i]:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"LSTM Validation ROC Curves (Macro AUC={macro_auc:.3f})")
plt.legend()
os.makedirs("plots", exist_ok=True)
roc_path = "plots/lstm_ROC_curves.png"
plt.savefig(roc_path, bbox_inches="tight")
plt.close()
print(f"ROC curves saved to {roc_path}")

# -------------------------------------------------------------
# 6. Confusion Matrix
# -------------------------------------------------------------
print("Generating confusion matrix...")

cm = confusion_matrix(y_val, preds, normalize="true")
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(num_classes)],
            yticklabels=[f"Class {i}" for i in range(num_classes)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (Validation Set)")
cm_path = "plots/lstm_confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# -------------------------------------------------------------
# 7. t-SNE Visualization
# -------------------------------------------------------------
print("Generating t-SNE visualization...")
embeddings = X_val.reshape(X_val.shape[0], -1)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.scatter(X_tsne[y_val == i, 0], X_tsne[y_val == i, 1], label=f"Class {i}", s=15)
plt.title("t-SNE Visualization of LSTM Feature Space (Validation Data)")
plt.legend()
tsne_path = "plots/lstm_TSNE_features.png"
plt.savefig(tsne_path, bbox_inches="tight")
plt.close()
print(f"t-SNE visualization saved to {tsne_path}")

# -------------------------------------------------------------
# 8. Per-Class Metric Comparison Chart
# -------------------------------------------------------------
print("Generating per-class Precision, Recall, F1 bar chart...")
metrics_df = pd.DataFrame({
    "Class": [f"Class {i}" for i in range(num_classes)],
    "Precision": precision_per_class,
    "Recall": recall_per_class,
    "F1-Score": f1_per_class
})

plt.figure(figsize=(9, 6))
metrics_df.plot(x="Class", kind="bar", ax=plt.gca())
plt.title("Per-Class Precision, Recall, and F1 Scores")
plt.xlabel("Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
bar_path = "plots/lstm_classwise_metrics.png"
plt.savefig(bar_path, bbox_inches="tight")
plt.close()
print(f"Per-class metrics bar chart saved to {bar_path}")

# -------------------------------------------------------------
# 9. Save all metrics to CSV
# -------------------------------------------------------------
os.makedirs("results", exist_ok=True)
metrics_path = "results/metrics_lstm_val.csv"

metrics = {
    "Accuracy": [acc],
    "Weighted_F1": [f1_weighted],
    "Weighted_Precision": [precision_weighted],
    "Weighted_Recall": [recall_weighted],
    "Macro_AUC": [macro_auc],
    "Micro_AUC": [micro_auc]
}

for i in range(num_classes):
    metrics[f"AUC_Class_{i}"] = [roc_auc[i]]
    metrics[f"Precision_Class_{i}"] = [precision_per_class[i]]
    metrics[f"Recall_Class_{i}"] = [recall_per_class[i]]
    metrics[f"F1_Class_{i}"] = [f1_per_class[i]]

pd.DataFrame(metrics).to_csv(metrics_path, index=False)
print(f"\nMetrics saved to {metrics_path}")

print("\n Evaluation complete! Files saved:")
print(f" - ROC curves: {roc_path}")
print(f" - Confusion matrix: {cm_path}")
print(f" - t-SNE: {tsne_path}")
print(f" - Classwise metrics bar chart: {bar_path}")
print(f" - Metrics CSV: {metrics_path}")
