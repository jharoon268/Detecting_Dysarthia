import torch
import torch.nn as nn

class LSTMSpectrogramClassifier(nn.Module):
    def __init__(self, n_mels=128, hidden_size=256, num_layers=3, num_classes=5, dropout=0.4, bidirectional=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        # Instead of hardcoding conv_feat_dim, compute dynamically
        with torch.no_grad():
            sample = torch.zeros(1, 1, n_mels, 754)  # (batch, channel, mel, time)
            out = self.conv(sample)
            _, c_out, mel_out, time_out = out.shape
            self.conv_feat_dim = c_out * mel_out
            print(f"[Model Init] LSTM input feature size: {self.conv_feat_dim}")

        self.lstm = nn.LSTM(
            input_size=self.conv_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # x: (batch, seq_len, n_mels)
        b, seq_len, n_mels = x.size()
        x = x.unsqueeze(1)  # (b, 1, seq_len, n_mels)
        x = x.permute(0, 1, 3, 2)  # (b, 1, n_mels, seq_len)
        x = self.conv(x)  # (b, C, mel', seq_len')
        x = x.permute(0, 3, 1, 2)  # (b, seq_len', C, mel')
        b, seq_len_p, C, mel_p = x.shape
        x = x.reshape(b, seq_len_p, C * mel_p)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.classifier(x)
