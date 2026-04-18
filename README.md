# Dysarthria Detection using Whisper, Wav2Vec2, CNN & LSTM

A comprehensive deep learning project for **dysarthria severity classification** from speech audio. Four baseline models (CNN+MFCC, LSTM+Spectrogram, Wav2Vec2-base, Whisper-tiny) were implemented and compared, with a proposed fine-tuned Wav2Vec2 approach achieving incremental improvements.

**Course:** CS-272 Artificial Intelligence  
**Instructor:** Dr. Mehwish Fatima  
**Semester:** Fall 2025  
**Institution:** National University of Sciences and Technology (NUST)

---

## Team Members & Contributions

| Student | Baseline / Model | Responsibilities |
|---------|------------------|------------------|
| **Ayesha Kamran** | CNN + MFCC | Data preprocessing, CNN implementation |
| **Muskan Ejaz** | LSTM + Spectrogram | Feature extraction, LSTM model training |
| **Zainab Fatima** | Wav2Vec2-base | Model fine-tuning, evaluation metrics |
| **Juwairiya Haroon** | Whisper-tiny | Whisper fine-tuning, experimental infrastructure, hyperparameter optimization, pipeline integration |

Each team member developed, trained, and evaluated one baseline independently while sharing data preprocessing and evaluation scripts collaboratively.

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.0.1, Hugging Face Transformers 4.35.0 |
| Audio Processing | Librosa, torchaudio, MFCC, Spectrograms |
| Models | Whisper-tiny, Wav2Vec2-base, CNN, LSTM |
| Hardware | NVIDIA T4 / Tesla K80 (Google Colab Pro) |
| Evaluation | Accuracy, F1-score, Confusion Matrix, ROC Curves |

---

## Dataset

- **SAND Dysarthria Corpus** – 2,176 training samples
- **5 severity levels** (0 = healthy, 1 = mild, 2 = moderate, 3 = severe, 4 = profound)
- **Audio format:** 16kHz mono channel, normalized to [-1, 1] range
- **Data split:** 80/20 stratified train/validation (maintaining class distribution)

---

## Pipeline Overview
Raw Audio (.wav)
↓
Preprocessing (resample 16kHz, normalize)
↓
Feature Extraction (MFCC / Spectrogram / Raw Waveform)
↓
Model Training (CNN / LSTM / Wav2Vec2 / Whisper)
↓
Evaluation (Accuracy, F1, Confusion Matrix)
↓
Result Analysis & Comparison


### Pipeline Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads .wav files, splits into train/validation/test |
| `preprocess.py` | Extracts MFCCs/spectrograms, normalizes, pads sequences |
| `model_baseline_X.py` | Defines baseline architectures (CNN, LSTM, Wav2Vec2, Whisper) |
| `train.py` | Trains models with defined hyperparameters |
| `evaluate.py` | Tests performance, computes metrics (Accuracy, F1, Confusion Matrix) |

---

## Baseline Architectures

### Baseline A: CNN + MFCC
- Two convolutional layers + max pooling + dense layers + softmax output
- **Hyperparameters:** LR=0.001, Batch=32, Epochs=30, Optimizer=Adam
- **Rationale:** CNN captures local frequency-time features from MFCCs efficiently

### Baseline B: LSTM + Spectrogram
- Converts audio to time-frequency spectrograms → stacked LSTM layers
- **Hyperparameters:** LR=1e-3, Batch=16, Epochs=10, Optimizer=Adam
- **Rationale:** LSTM models temporal dependencies across speech frames

### Baseline C: Wav2Vec2-base
- Frozen Wav2Vec2 encoder (768-dim features) + FC layers with dropout
- **Hyperparameters:** LR=1e-3, Batch=64, Epochs=20, Optimizer=Adam
- **Rationale:** Leverages self-supervised pretraining on raw waveforms

### Baseline D: Whisper-tiny
- Compact encoder-decoder Transformer for audio feature processing
- **Hyperparameters:** LR=2e-4, Batch=8, Epochs=5, Optimizer=Adam
- **Rationale:** Robust generalization on small datasets

---

## Proposed Method: Fine-Tuned Wav2Vec2

Building on Baseline C, we implemented a strategic fine-tuning approach:

- **Gradual unfreezing** of transformer layers
- **Differential learning rates** (8e-4 for classification head, lower for encoder)
- **Enhanced regularization:** Dropout (0.2-0.4), Weight Decay (1e-3), Gradient Clipping (0.5)
- **Custom classification head** with batch normalization and layer-specific dropout

---

## Results

### Model Comparison

| Model | Input Features | Accuracy | Macro F1 | Train Time |
|-------|---------------|----------|----------|------------|
| CNN + MFCC | MFCC | 78% | 0.75 | 15 min |
| LSTM + Spectrogram | Spectrogram | 80% | 0.77 | 20 min |
| Wav2Vec2-base | Raw waveform | 83% | 0.81 | 25 min |
| **Whisper-tiny** | Raw waveform | **84%** | **0.83** | 30 min |

**Observation:** Whisper-tiny achieved the best accuracy and generalization, followed closely by Wav2Vec2-base.

### Fine-Tuned Wav2Vec2 Improvements

| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| Final Training Accuracy | 93.51% | 96.03% | **+2.53%** |
| Final Validation Accuracy | 46.56% | 49.08% | **+2.52%** |
| Best Validation Accuracy | 50.69% | 51.83% | **+1.15%** |

---

## How to Run (With Access to SAND Dataset)

### 1. Clone the repository
git clone https://github.com/jharoon268/Detecting_Dysarthia.git
cd Detecting_Dysarthia

### 2. Install Dependencies
pip install torch torchaudio transformers datasets librosa soundfile numpy scikit-learn matplotlib seaborn

### 3. Prepare Dataset
Place SAND dataset in data/ folder with the following structure:

data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   ├── class_2/
│   ├── class_3/
│   └── class_4/
└── test/
    ├── class_0/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── class_4/

### 4. Run training (example for Whisper-tiny)
python train.py --model whisper --epochs 5 --batch_size 8 --lr 2e-4

### 5. Run evaluation
python evaluate.py --model whisper --checkpoint checkpoints/whisper_best.pt

---

## Implementation Notes & Challenges
Challenge |	Solution
Large transformer models (Wav2Vec2, Whisper) require high memory	 | GPU acceleration (NVIDIA T4 on Google Colab Pro)
Audio preprocessing (MFCC/Spectrogram extraction) time-consuming | Optimized batch processing with Librosa
Noisy or truncated audio samples | Filtering and validation before training
Overfitting on small dataset (2,176 samples) | Dropout, weight decay, early stopping, gradient clipping

--- 

## Future Work
Data augmentation – Speed perturbation, noise addition tailored to dysarthric speech

Multi-modal integration – Combine acoustic features with articulatory or textual data

Cross-corpus evaluation – Validate on multiple dysarthria datasets

Clinical deployment – Real-time assessment tools for clinical workflows

Explainability – Attention visualization for clinician interpretability
