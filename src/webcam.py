"""Real-time webcam detection."""

import cv2
import argparse
from pathlib import Path
from time import time
import sys

from .inference import GestureNumberDetector
from .utils import draw_detections


class WebcamDetector:
    """Real-time webcam gesture and number detector."""
    
    def __init__(self, model_path: str, camera_id: int = 0, confidence_threshold: float = 0.5):
        self.detector = GestureNumberDetector(model_path, confidence_threshold=confidence_threshold)
        self.camera_id = camera_id
    
    def run(self, show_fps: bool = True, show_stats: bool = True):
        """Run real-time detection from webcam."""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        start_time = time()
        gesture_frames = 0
        number_67_frames = 0
        
        print("\n67-FINDER: Real-time Gesture & Number 67 Detection")
        print("Controls: Q - Quit, S - Screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result = self.detector.detect_frame(frame)
            frame_with_detections = draw_detections(frame, result["detections"])
            
            if result["has_gesture"]:
                gesture_frames += 1
            if result["has_67"]:
                number_67_frames += 1
            
            if show_fps:
                fps_text = f"FPS: {frame_count / (time() - start_time):.1f}"
                cv2.putText(frame_with_detections, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            status = f"Gesture: {result['has_gesture']} | 67: {result['has_67']}"
            cv2.putText(frame_with_detections, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('67-Finder', frame_with_detections)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == ord('s') or key == ord('S'):
                cv2.imwrite(f"screenshot_{frame_count}.jpg", frame_with_detections)
                print(f"Screenshot saved: screenshot_{frame_count}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames: {frame_count}")
        print(f"Frames with gesture: {gesture_frames}")
        print(f"Frames with 67: {number_67_frames}")


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture and number 67 detection")
    parser.add_argument("--model", type=str, default="models/pretrained/best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    
    detector = WebcamDetector(model_path=args.model, camera_id=args.camera, confidence_threshold=args.confidence)
    detector.run()


if __name__ == "__main__":
    main()
