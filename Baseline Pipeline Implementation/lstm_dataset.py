import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, X_path, y_path=None):
        self.X = np.load(X_path)
        self.y = None
        if y_path is not None:
            self.y = np.load(y_path)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].astype('float32')  # (time, features)
        if self.y is None:
            return x
        y = int(self.y[idx])
        return x, y
