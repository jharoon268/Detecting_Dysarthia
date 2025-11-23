import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

def extract_features(file_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # --- Core features ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

        # Combine all features along axis 0
        features = np.vstack([mfcc, chroma, contrast, tonnetz, zcr, rms])
        return features.T

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def normalize_and_pad(features_list, max_len=None):
    if not features_list:
        raise ValueError("No features to normalize or pad!")

    if not max_len:
        max_len = max(f.shape[0] for f in features_list if f is not None)

    scaler = StandardScaler()
    padded = []
    for f in features_list:
        if f is None:
            continue
        f = scaler.fit_transform(f)
        if f.shape[0] < max_len:
            pad = max_len - f.shape[0]
            f = np.pad(f, ((0, pad), (0, 0)), mode='constant')
        else:
            f = f[:max_len, :]
        padded.append(f)
    return np.array(padded)


def preprocess_dataset(csv_path, save_dir='results', is_train=True):
    df = pd.read_csv(csv_path)
    features_list, labels = [], []

    print(f"\nExtracting features from {len(df)} audio files...")

    for _, row in df.iterrows():
        fpath = row['file_path']
        feat = extract_features(fpath)
        if feat is not None:
            features_list.append(feat)
            if is_train:
                labels.append(row.get('class', None))

    if not features_list:
        raise ValueError("No valid audio features were extracted!")

    print("Feature extraction done. Normalizing and padding...")
    X = normalize_and_pad(features_list)
    os.makedirs(save_dir, exist_ok=True)

    if is_train:
        y = np.array(labels, dtype=np.float32)
        np.save(os.path.join(save_dir, 'X_train.npy'), X)
        np.save(os.path.join(save_dir, 'y_train.npy'), y)
        print(f" Saved X_train.npy and y_train.npy in {save_dir}")
        return X, y
    else:
        np.save(os.path.join(save_dir, 'X_test.npy'), X)
        print(f" Saved X_test.npy in {save_dir}")
        return X, None


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    out_dir = os.path.join(base_dir, "results")

    train_csv = os.path.join(base_dir, "results", "train_data.csv")
    test_csv = os.path.join(base_dir, "results", "test_data.csv")

    preprocess_dataset(train_csv, out_dir, True)
    preprocess_dataset(test_csv, out_dir, False)
