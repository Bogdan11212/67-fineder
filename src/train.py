"""Training script for gesture and number detection model."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    model_name: str = "yolov8m"
):
    """Train YOLOv8 model."""
    
    # Load model
    model = YOLO(f"{model_name}.pt")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=20,
        save=True,
        project="runs/detect",
        name="67-finder",
        exist_ok=True,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    print(f"Training complete! Results saved in runs/detect/67-finder/")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture and number detector")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--model", type=str, default="yolov8m", help="Model size (n, s, m, l, x)")
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model
    )
