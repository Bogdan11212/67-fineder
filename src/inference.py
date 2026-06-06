"""Inference engine for gesture and number detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO
import torch

from .utils import draw_detections


class GestureNumberDetector:
    """Real-time gesture and number 67 detector using YOLOv8."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = None):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(device)
        
        self.class_names = {0: "Gesture", 1: "Number 67"}
        self.results_history = []
    
    def detect_image(self, image_path: str) -> Dict:
        """Detect in single image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect_frame(image, image_path)
    
    def detect_frame(self, frame: np.ndarray, source_id: str = "frame") -> Dict:
        """Detect in single frame."""
        results = self.model(frame, conf=self.confidence_threshold, device=self.device, verbose=False)
        detections = self._parse_results(results[0])
        
        output = {
            "source": source_id,
            "frame_shape": frame.shape,
            "detections": detections,
            "has_gesture": any(d['class_id'] == 0 for d in detections),
            "has_67": any(d['class_id'] == 1 for d in detections),
        }
        
        self.results_history.append(output)
        return output
    
    def _parse_results(self, result) -> List[Dict]:
        """Parse YOLOv8 results."""
        detections = []
        if result.boxes is None:
            return detections
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "class_name": self.class_names.get(cls, "Unknown"),
            }
            detections.append(detection)
        
        return detections
