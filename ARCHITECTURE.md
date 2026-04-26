# ERVS System Architecture

## Data Flow
Video File → preprocessing.py (resize 640x640) → inference.py (ONNX model) → utils.py (draw boxes, heatmap, tripwire, dwell) → logger.py (log events) → main.py (display frame)

## Files
- main.py: Entry point, video loop, display
- preprocessing.py: Frame resize and normalization
- inference.py: Load ONNX model, run detection, return boxes + scores + FPS
- utils.py: Draw boxes, tripwire counter, heatmap array, dwell time zones
- config.py: All settings (paths, thresholds, line coords, zone coords)
- logger.py: Write events to .log file
- training.py: Train YOLOv8n and YOLOv5n, export to ONNX
- benchmark.py: Compare FPS and mAP across 3 models

## Models Benchmarked
1. YOLOv8n - primary candidate
2. YOLOv5n - legacy candidate  
3. SSD-MobileNet V2 - mobile-first candidate

## Target Hardware
- Training: Laptop (GTX 1650 Ti, CUDA 11.8)
- Deployment: NVIDIA Jetson Nano 4GB (ONNX Runtime)

## Input/Output
- Input: MP4 video file or live camera
- Output: Annotated video with bounding boxes, FPS, footfall count, heatmap overlay