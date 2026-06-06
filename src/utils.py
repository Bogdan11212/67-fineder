"""Utility functions for 67-Finder."""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def load_model(model_path: str):
    """Load YOLOv8 model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        YOLO model object
    """
    from ultralytics import YOLO
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    return model


def save_results(results: Dict, output_dir: str = "results") -> str:
    """Save detection results to JSON.
    
    Args:
        results: Detection results dictionary
        output_dir: Directory to save results
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_file


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """Draw bounding boxes on image.
    
    Args:
        image: Input image (numpy array)
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence score
        
    Returns:
        Image with drawn detections
    """
    class_colors = {
        0: (0, 255, 0),    # Gesture - Green
        1: (0, 0, 255),    # Number 67 - Red
    }
    
    class_names = {0: "Gesture", 1: "Number 67"}
    
    result = image.copy()
    
    for det in detections:
        if det.get('confidence', 0) < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_id = det.get('class_id', 0)
        confidence = det.get('confidence', 0)
        
        color = class_colors.get(class_id, (255, 255, 0))
        label = f"{class_names.get(class_id, 'Unknown')} {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            result,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        cv2.putText(
            result,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return result


def get_fps(start_time: float, frame_count: int) -> float:
    """Calculate FPS.
    
    Args:
        start_time: Start time
        frame_count: Number of frames
        
    Returns:
        Frames per second
    """
    from time import time
    elapsed = time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0


def create_output_directory(base_path: str = "output") -> str:
    """Create timestamped output directory.
    
    Args:
        base_path: Base output path
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_image(image_path: str) -> bool:
    """Validate if image file exists and is readable.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        return img is not None
    except Exception:
        return False
