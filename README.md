# 🔬 Cancer Image Classification — InceptionV3

## 📌 Overview

A **PyTorch transfer-learning pipeline** that classifies breast-ultrasound images from the
[BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) into three categories:

| Label | Class |
|-------|-------|
| 0 | Benign |
| 1 | Malignant |
| 2 | Normal |

The backbone is **InceptionV3** pretrained on ImageNet, fine-tuned for the 3-class task.

---

## 🧠 Tech Stack

| Library | Role |
|---------|------|
| Python 3 | Language |
| PyTorch | Model training & dataloading |
| TorchVision | Pretrained model + transforms |
| NumPy | Array utilities |
| Matplotlib | Visualisation |

---

## 📊 Dataset — BUSI

- **Source:** Breast Ultrasound Images Dataset (BUSI)
- **Classes:** benign · malignant · normal
- Each scan ships with a paired `_mask` PNG → masks are **filtered out** before training
- Split: **80 % train / 20 % test** (index-based, deterministic)

---

## 🔄 Data Pipeline

```
Raw ImageFolder
      │
      ▼
Filter mask images          (keep only original scans)
      │
      ▼
80 / 20 index split
      │
      ├─► train_ds  ──► TRAIN_TRANSFORMS  (resize + jitter + blur + crop + tensor)
      │
      └─► test_ds   ──► TEST_TRANSFORMS   (resize + tensor only — no augmentation)
      │
      ▼
DataLoader (shuffle=True for train, False for test)
```

> **Why two `ImageFolder` calls?**  
> `ImageFolder` stores a single transform for the whole dataset.  
> Loading the folder twice — each with its own `Compose` — and assigning disjoint  
> index slices via `Subset` is the cleanest way to guarantee augmentation is applied  
> **only** to the training split.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input size | 299 × 299 (InceptionV3 requirement) |
| Batch size | 20 |
| Epochs | 10 |
| Optimiser | AdamW |
| Classes | 3 |

---

## 🚀 Progress

- [x] Project setup
- [x] Library imports
- [x] Constants & transforms definition
- [x] Dataset loading & random image visualisation
- [x] Mask-image filtering
- [x] Train / test split with correct per-split transforms
- [x] DataLoader construction
- [x] Batch visualisation
- [x] Model definition (InceptionV3 fine-tuning)
- [x] Training loop
- [ ] Evaluation & metrics
- [ ] Deployment

---

## 📈 Results

*(To be updated after training)*

---

## 📁 Project Structure

```
├── Cancer_Image_Classification.ipynb   # Main notebook
└── README.md                           # This file
```