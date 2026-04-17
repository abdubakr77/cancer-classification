# 🔬 Cancer Image Classification — VGG19-BN

## 📌 Overview

A **PyTorch transfer-learning pipeline** that classifies breast-ultrasound images from the
[BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) into three categories:

| Label | Class |
|-------|-------|
| 0 | Benign |
| 1 | Malignant |
| 2 | Normal |

The backbone is **VGG19 with Batch Normalisation** pretrained on ImageNet.
The final fully-connected layer is replaced to output 3 logits; all other weights are fine-tuned end-to-end.

---

## 🧠 Tech Stack

| Library | Role |
|---------|------|
| Python 3 | Language |
| PyTorch | Model training & dataloading |
| TorchVision | Pretrained VGG19-BN + transforms |
| scikit-learn | Confusion matrix & classification report |
| NumPy | Array utilities & index shuffling |
| Matplotlib / Seaborn | Visualisation |
| tqdm | Training progress bars |

---

## 📊 Dataset — BUSI

- **Source:** Breast Ultrasound Images Dataset (BUSI)
- **Classes:** benign · malignant · normal
- Each scan ships with a paired `_mask` PNG → **masks are filtered out** before training
- Indices are **shuffled before splitting** to ensure class balance across both splits
- Split: **80 % train / 20 % test** (index-based, disjoint)

---

## 🔄 Data Pipeline

```
Raw ImageFolder (no transforms)
        │
        ▼
Filter mask images          keep paths where "mask" not in path
        │
        ▼
Shuffle valid indices       ensure class balance across splits
        │
        ▼
80 / 20 index split
        │
        ├─► train_ds ──► TRAIN_TRANSFORMS
        │                 Resize(224) · ColorJitter · GaussianBlur
        │                 RandomHorizontalFlip · RandomVerticalFlip
        │                 RandomRotation(15) · RandomAffine
        │                 ToTensor · Normalize(ImageNet stats)
        │
        └─► test_ds  ──► TEST_TRANSFORMS
                          Resize(224) · ToTensor · Normalize(ImageNet stats)
        │
        ▼
DataLoader
  train: shuffle=True  · batch=12
  test:  shuffle=False · batch=12
```

> **Why two `ImageFolder` calls?**
> `ImageFolder` stores a single transform for the whole dataset.
> Loading the same folder twice — each with its own `Compose` — and slicing
> disjoint indices via `Subset` guarantees augmentation is applied **only** to
> the training split, never to the test split.

---

## 🏗️ Model Architecture

```
VGG19-BN (ImageNet pretrained)
    └── features          16 conv blocks + BN + MaxPool
    └── avgpool           AdaptiveAvgPool2d
    └── classifier
            Linear(25088 → 4096) · ReLU · Dropout
            Linear(4096  → 4096) · ReLU · Dropout
            Linear(4096  → 3)    ← replaced for 3-class output
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input size | 224 × 224 | VGG19 requirement |
| Batch size | 12 | Tuned for GPU memory |
| Max epochs | 50 | with early stopping |
| Optimiser | AdamW | weight decay included |
| Learning rate | 1e-5 | low LR essential for fine-tuning |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=3 |
| Early stopping patience | 5 | stops if val loss stagnates |
| Classes | 3 | benign · malignant · normal |

---

## 🛠️ Key Design Decisions

**ImageNet Normalisation**
VGG19 was pretrained on ImageNet using specific mean and std values per channel.
Applying the same normalisation at inference time aligns the input distribution with
what the model learned during pretraining, which significantly improves transfer performance.

**Inverse-Frequency Class Weighting**
BUSI is imbalanced — benign cases outnumber malignant and normal ones.
Using unweighted cross-entropy causes the model to bias predictions toward the majority class.
Weighting the loss by the inverse of each class's sample count forces equal attention across all three classes.
The weights are computed from the training split only, never from the test split.

**Weighted Loss in Training, Unweighted in Evaluation**
The weighted loss is used during training to guide learning.
During evaluation, unweighted cross-entropy is used so that the reported loss reflects true model performance rather than a weighted proxy.

**LR Scheduling**
A fixed learning rate that works well early in training can overshoot the optimum in later epochs.
`ReduceLROnPlateau` halves the learning rate whenever validation loss stagnates for 3 consecutive epochs, allowing finer convergence without manual intervention.

**Augmentation Strategy**
BUSI contains approximately 780 valid scans — a small dataset for deep learning.
Augmentations such as flips, rotation, affine shifts, and colour jitter increase the effective diversity of training samples and help prevent overfitting.
All augmentations are clinically plausible for ultrasound imaging.

---

## 🚀 Progress

- [x] Project setup
- [x] Library imports
- [x] Constants & transforms definition (with ImageNet normalisation)
- [x] Dataset loading & random image visualisation
- [x] Class distribution inspection
- [x] Mask-image filtering
- [x] Index shuffle before split (fixes class imbalance in splits)
- [x] Train / test split with correct per-split transforms
- [x] DataLoader construction
- [x] Batch visualisation
- [x] GPU / CPU device detection
- [x] Inverse-frequency class weight computation
- [x] VGG19-BN model definition & head replacement
- [x] AdamW optimiser
- [x] ReduceLROnPlateau scheduler
- [x] Training loop with weighted loss, early stopping & best-model checkpointing
- [x] Learning curves (accuracy & loss)
- [x] Confusion matrix & classification report
- [x] Test-set prediction grid (green = correct, red = wrong)

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92% |
| Best Val Accuracy | 94.87% (Epoch 9) |
| Epochs Trained | 17 (early stopping) |

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 0.97 | 0.91 | 0.94 |
| Malignant | 0.87 | 0.93 | 0.90 |
| Normal | 0.86 | 0.95 | 0.90 |
| **Weighted Avg** | **0.93** | **0.92** | **0.92** |

---

## 📁 Project Structure

```
├── Cancer_Image_Classification.ipynb   # Main notebook
├── best_model.pth                      # Best checkpoint (saved during training)
└── README.md                           # This file
```