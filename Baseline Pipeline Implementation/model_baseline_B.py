# import torch
# import torch.nn as nn

# class LSTMSpectrogramClassifier(nn.Module):
#     def __init__(self, n_mels=128, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3, bidirectional=True):
#         super().__init__()

#         # --- Convolutional front-end ---
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d((1, 2)),

#             nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d((1, 2))
#         )

#         # Temporary conv output dimension placeholder (will be inferred in forward)
#         self.lstm = None
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.bidirectional = bidirectional
#         self.num_classes = num_classes

#         self.classifier = None  # will also be initialized dynamically later

#     def _init_lstm_and_classifier(self, sample_tensor):
#         """Initialize LSTM and classifier dynamically using real input."""
#         with torch.no_grad():
#             b, seq_len, n_mels = sample_tensor.size()
#             x = sample_tensor.unsqueeze(1).permute(0, 1, 3, 2)
#             x = self.conv(x)
#             x = x.permute(0, 3, 1, 2)
#             b, seq_len_p, C, mel_p = x.shape
#             input_dim = C * mel_p

#             # Define LSTM now that input size is known
#             self.lstm = nn.LSTM(
#                 input_size=input_dim,
#                 hidden_size=self.hidden_size,
#                 num_layers=self.num_layers,
#                 batch_first=True,
#                 bidirectional=self.bidirectional,
#                 dropout=self.dropout if self.num_layers > 1 else 0.0
#             )

#             lstm_output_dim = self.hidden_size * (2 if self.bidirectional else 1)
#             self.classifier = nn.Sequential(
#                 nn.Dropout(self.dropout),
#                 nn.Linear(lstm_output_dim, 64),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout),
#                 nn.Linear(64, self.num_classes)
#             )

#     def forward(self, x):
#         # If first forward call, build LSTM + classifier dynamically
#         if self.lstm is None:
#             self._init_lstm_and_classifier(x.to(next(self.conv.parameters()).device))

#         b, seq_len, n_mels = x.size()
#         x_conv = x.unsqueeze(1)             # (b, 1, seq_len, n_mels)
#         x_conv = x_conv.permute(0, 1, 3, 2) # (b, 1, n_mels, seq)
#         x_conv = self.conv(x_conv)
#         x_conv = x_conv.permute(0, 3, 1, 2) # (b, seq', C, mel')
#         b, seq_len_p, C, mel_p = x_conv.shape
#         x_conv = x_conv.reshape(b, seq_len_p, C * mel_p)
#         out, _ = self.lstm(x_conv)
#         pooled = out.mean(dim=1)
#         logits = self.classifier(pooled)
#         return logits

# # Debug quick check
# if __name__ == "__main__":
#     model = LSTMSpectrogramClassifier(n_mels=128)
#     dummy = torch.randn(2, 754, 128)
#     out = model(dummy)
#     print(" Output shape:", out.shape)



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
