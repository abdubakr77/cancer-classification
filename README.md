# 🔬 Cancer Image Classification — VGG16-BN

## 📌 Overview

A **PyTorch transfer-learning pipeline** that classifies breast-ultrasound images from the
[BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) into three categories:

| Label | Class |
|-------|-------|
| 0 | Benign |
| 1 | Malignant |
| 2 | Normal |

The backbone is **VGG16 with Batch Normalisation** pretrained on ImageNet.  
Only the final fully-connected layer is replaced to output 3 logits; all other weights are fine-tuned.

---

## 🧠 Tech Stack

| Library | Role |
|---------|------|
| Python 3 | Language |
| PyTorch | Model training & dataloading |
| TorchVision | Pretrained VGG16-BN + transforms |
| NumPy | Array utilities & index shuffling |
| Matplotlib | Visualisation |
| tqdm | Training progress bars |

---

## 📊 Dataset — BUSI

- **Source:** Breast Ultrasound Images Dataset (BUSI)
- **Classes:** benign · malignant · normal
- Each scan ships with a paired `_mask` PNG → **masks are filtered out** before training
- Indices are **shuffled before splitting** to ensure class balance across both splits
- Split: **80 % train / 20 % test** (index-based)

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
        │                 RandomHorizontalFlip · ToTensor
        │
        └─► test_ds  ──► TEST_TRANSFORMS
                          Resize(224) · ToTensor only
        │
        ▼
DataLoader
  train: shuffle=True  · batch=20
  test:  shuffle=False · batch=20
```

> **Why two `ImageFolder` calls?**  
> `ImageFolder` stores a single transform for the whole dataset.  
> Loading the same folder twice — each with its own `Compose` — and slicing  
> disjoint indices via `Subset` guarantees augmentation is applied **only** to  
> the training split, never to the test split.

---

## 🏗️ Model Architecture

```
VGG16-BN (ImageNet pretrained)
    └── features          13 conv blocks + BN + MaxPool  [frozen pretrained]
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
| Input size | 224 × 224 | VGG16 requirement |
| Batch size | 20 | |
| Max epochs | 20 | with early stopping |
| Optimiser | AdamW | |
| Learning rate | 1e-5 | low LR essential for fine-tuning |
| Early stopping patience | 3 | stops if val loss stagnates |
| Classes | 3 | benign · malignant · normal |

---

## 🚀 Progress

- [x] Project setup
- [x] Library imports
- [x] Constants & transforms definition
- [x] Dataset loading & random image visualisation
- [x] Mask-image filtering
- [x] Index shuffle before split (fixes class imbalance in splits)
- [x] Train / test split with correct per-split transforms
- [x] DataLoader construction
- [x] Batch visualisation
- [x] VGG16-BN model definition & head replacement
- [x] AdamW optimiser
- [x] Training loop with early stopping & best-model checkpointing
- [x] Learning curves (accuracy & loss)
- [x] Test-set prediction grid (green = correct, red = wrong)
- [ ] Confusion matrix & classification report
- [ ] Deployment

---

## 📈 Results

*(To be updated after training)*

---

## 📁 Project Structure

```
├── Cancer_Image_Classification.ipynb   # Main notebook
├── best_model.pth                      # Best checkpoint (saved during training)
└── README.md                           # This file
```