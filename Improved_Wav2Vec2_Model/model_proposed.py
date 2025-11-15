
import torch
import torch.nn as nn

class BaselineClassifier(nn.Module):
    """Baseline model with strong regularization"""
    def __init__(self, input_dim=768, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x): 
        return self.classifier(x)

class ProposedDysarthriaClassifier(nn.Module):
    """Enhanced classifier with balanced architecture"""
    def __init__(self, input_dim=768, num_classes=5):
        super().__init__()
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
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
        return self.enhanced_classifier(x)
