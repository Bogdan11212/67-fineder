# Dataset Collection Guide

## Directory Structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

## Classes

- `0`: Gesture (hand gestures)
- `1`: Number 67

## Collecting Training Data

### Method 1: Bing Image Downloader (Automated)

```bash
pip install bing-image-downloader

python data/scripts/download_dataset.py
```

### Method 2: Manual Collection

1. Create folders: `data/images/train`, `data/images/val`, `data/images/test`
2. Add your images
3. Label using LabelImg or Roboflow

### YOLO Format

Annotations must be in YOLO format (`.txt` files):

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]

### Dataset Split

- Training: 70% of data
- Validation: 20% of data
- Test: 10% of data

## Tools for Labeling

- **Roboflow** (recommended): https://roboflow.com
- **LabelImg**: https://github.com/hartleygriffin/labelImg
- **CVAT**: https://cvat.org/
