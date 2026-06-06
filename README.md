# 67-Finder: Gesture & Number Detection in Real-Time

🎯 **A real-time detector for hand gestures and the number 67 using YOLOv8**

## Features

✨ **Real-time Detection:**
- 🤖 Detect hand gestures from webcam live stream
- 🔢 Detect number "67" in images and video
- ⚡ GPU-optimized for fast inference
- 🖼️ Process both images and video streams

📊 **Training & Data:**
- 🤖 Automated dataset collection
- 📈 Data augmentation pipeline
- 🔄 Automatic training via GitHub Actions
- ☁️ Model storage on Hugging Face Hub

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Bogdan11212/67-fineder.git
cd 67-fineder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Real-time Webcam Detection
```bash
python src/webcam.py --model models/pretrained/best.pt
```

#### 2. Image Detection
```bash
python src/inference.py --image path/to/image.jpg --model models/pretrained/best.pt
```

#### 3. Video Detection
```bash
python src/inference.py --video path/to/video.mp4 --model models/pretrained/best.pt
```

## Project Structure

```
67-fineder/
├── .github/workflows/
│   ├── train.yml              # Training pipeline
│   ├── test.yml               # Testing pipeline
│   └── deploy.yml             # Deploy to Hugging Face
├── data/
│   ├── scripts/
│   │   ├── download_dataset.py    # Download training data
│   │   ├── augment_data.py        # Data augmentation
│   │   └── prepare_dataset.py     # Dataset preparation
│   ├── images/                # Training images
│   ├── annotations/           # YOLO format annotations
│   └── README.md              # Data collection guide
├── models/
│   ├── config.yaml            # Model configuration
│   └── pretrained/            # Trained models
├── src/
│   ├── train.py               # Training script
│   ├── inference.py           # Image/Video inference
│   ├── webcam.py              # Real-time webcam
│   ├── image_processor.py     # Image utilities
│   └── utils.py               # Helper functions
├── tests/
│   ├── test_inference.py      # Inference tests
│   └── test_webcam.py         # Webcam tests
├── notebooks/
│   └── demo.ipynb             # Demo notebook
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Dataset Preparation

### Collecting Training Data

1. **Using provided script:**
   ```bash
   python data/scripts/download_dataset.py
   ```

2. **Manual collection:**
   - Create `data/images/` directory
   - Place your images there
   - Use a labeling tool (Roboflow, LabelImg) to create annotations

3. **Data augmentation:**
   ```bash
   python data/scripts/augment_data.py
   ```

### YOLO Format

Annotations should be in YOLO format (`.txt` files):
```
<class_id> <x_center> <y_center> <width> <height>
```

Class IDs:
- `0` = Gesture
- `1` = Number 67

## Training

### Local Training

```bash
python src/train.py \
  --data data/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### Automatic Training (GitHub Actions)

The repository is configured to automatically:
1. Download/prepare dataset
2. Train model on GPU
3. Evaluate performance
4. Upload to Hugging Face Hub

Trigger with:
```bash
git push  # or manually via Actions tab
```

## Model Architecture

- **Base Model:** YOLOv8
- **Input Size:** 640x640 pixels
- **Output:** Bounding boxes + confidence scores
- **Classes:** 
  - Gesture (all hand gestures)
  - Number 67

## Real-time Performance

- ⚡ **FPS:** 30+ on GPU (RTX 3060+)
- ⚡ **FPS:** 15+ on CPU (i7+)
- 🎯 **Latency:** <50ms per frame (GPU)

## API Usage

```python
from src.inference import GestureNumberDetector

# Initialize detector
detector = GestureNumberDetector('models/pretrained/best.pt')

# Detect in image
results = detector.detect_image('image.jpg')
print(results)  # {'gestures': [...], 'numbers': [...]}

# Detect in video
detector.detect_video('video.mp4', output='output.mp4')
```

## Model Export

The trained model can be exported to various formats:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/pretrained/best.pt')
model.export(format='onnx')    # ONNX
model.export(format='tflite')  # TensorFlow Lite
model.export(format='engine')  # TensorRT
"
```

## Hugging Face Integration

Models are automatically uploaded to:
- 🤖 [67-finder on Hugging Face](https://huggingface.co/Bogdan11212/67-finder)

Use directly from Hugging Face:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Bogdan11212/67-finder",
    filename="best.pt"
)
```

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 4GB+ RAM
- Webcam or video input

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - See LICENSE file for details

## Contact

- GitHub: [@Bogdan11212](https://github.com/Bogdan11212)
- Issues: [GitHub Issues](https://github.com/Bogdan11212/67-fineder/issues)

---

**Made with ❤️ for real-time object detection**
