# RUN EXPERIMENTS
# FINAL ATTEMPT: BETTER MODELS AND TRAINING
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set paths
DRIVE_TASK1_PATH = "/content/drive/MyDrive/task1"

print("🚀 FINAL ATTEMPT: IMPROVING MODEL PERFORMANCE...")

# 1. Load and preprocess features
print("Loading features...")
data = torch.load(f"{DRIVE_TASK1_PATH}/wav2vec_features.pth")
features, labels = data['features'], data['labels']

# Check class distribution
unique, counts = np.unique(labels.numpy(), return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# Normalize features
scaler = StandardScaler()
features_np = features.numpy()
features_np = scaler.fit_transform(features_np)
features = torch.tensor(features_np, dtype=torch.float32)

print(f"✅ Loaded {len(features)} samples")

# 2. BETTER MODEL ARCHITECTURES

class ImprovedBaselineClassifier(nn.Module):
    """Improved baseline with better capacity"""
    def __init__(self, input_dim=768, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x): 
        return self.classifier(x)

class ImprovedProposedClassifier(nn.Module):
    """Proposed model with optimal capacity"""
    def __init__(self, input_dim=768, num_classes=5):
        super().__init__()
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x): 
        return self.enhanced_classifier(x)

# 3. OPTIMIZED training strategy
def train_model_optimized(model, features, labels, model_name, batch_size=32, epochs=80, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create datasets
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer with different settings
    if model_name == "Baseline":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # More patient scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0
    patience_counter = 0
    patience = 20  # More patience
    
    print(f"Training {model_name} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"{model_name} Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'{DRIVE_TASK1_PATH}/best_{model_name.lower()}_model.pth')
            print(f"🎯 New best {model_name} validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'{DRIVE_TASK1_PATH}/best_{model_name.lower()}_model.pth'))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'final_train_accuracy': train_accs[-1],
        'final_val_accuracy': val_accs[-1],
        'best_val_accuracy': best_val_acc
    }

# 4. Run OPTIMIZED experiments
print("\n📊 Running IMPROVED Baseline...")
baseline_results = train_model_optimized(
    ImprovedBaselineClassifier(), features, labels, "Baseline", 
    batch_size=32, epochs=80, lr=1e-3
)

print("\n📊 Running IMPROVED Proposed...")
proposed_results = train_model_optimized(
    ImprovedProposedClassifier(), features, labels, "Proposed",
    batch_size=32, epochs=80, lr=8e-4
)

# 5. Save results
results = {
    'baseline': baseline_results,
    'proposed': proposed_results
}

results_path = f'{DRIVE_TASK1_PATH}/experiment_results_improved.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Improved results saved to: {results_path}")
