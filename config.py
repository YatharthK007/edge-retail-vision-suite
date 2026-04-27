import os


# PATHS

# Root of the project (one level above this file).  Works regardless of where the script is launched from, as long as config.py lives at the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#  Model paths 
# PRIMARY deployment model (ONNX — runs on both laptop and Jetson Nano).
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.onnx")

# Individual candidate model paths (used by benchmark.py)
MODEL_YOLOV8N_ONNX = os.path.join(PROJECT_ROOT, "models", "yolov8n.onnx")
MODEL_YOLOV5N_ONNX  = os.path.join(PROJECT_ROOT, "models", "yolov5n.onnx")
MODEL_SSD_ONNX      = os.path.join(PROJECT_ROOT, "models", "ssd_mobilenet.onnx")

#  Input source 
# Swap this string to "0" (or 1, 2 …) to use a live USB camera instead.
INPUT_SOURCE = os.path.join(PROJECT_ROOT, "dataset", "test_video.mp4")

#  Output / logging 
LOG_FILE_PATH       = os.path.join(PROJECT_ROOT, "logs", "ervs_log.txt")
HEATMAP_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "logs", "heatmap.png")

# Automatically create the logs/ directory if it does not already exist so the rest of the codebase can write without checking first.
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)



# MODEL / INFERENCE SETTINGS

# Minimum confidence a detection must have to be kept.
CONFIDENCE_THRESHOLD = 0.5

# Intersection-over-Union threshold used during Non-Maximum Suppression.
NMS_THRESHOLD = 0.4

# Spatial dimensions fed to the model (width, height).  All three benchmarked architectures accept 640×640 inputs; SSD-MobileNet will be resized to match.
INPUT_SIZE = (640, 640)  # (width, height)

# COCO class index for "person".  Change to None to detect all classes.
PERSON_CLASS_ID = 0



# PERFORMANCE / FRAME PROCESSING

# Processing only every N-th frame to reduce CPU/GPU load on the Jetson Nano.
# Set to 1 to process every frame (maximum accuracy, higher load).
# Set to 2 to process every 2nd frame (good balance for Jetson Nano 4 GB).
FRAME_SKIP = 2



# HARDWARE / RUNTIME

# Set True to attempt CUDA acceleration on the laptop.
# On the Jetson Nano the ONNX Runtime selects the best available provider (CPU or TensorRT) automatically, this flag is used only as a hint.
USE_CUDA = True

# ONNX Runtime execution providers tried in order of preference.
# "CUDAExecutionProvider" is skipped automatically if CUDA is unavailable, so this list is safe for both laptop and Jetson Nano.
ONNX_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if USE_CUDA
    else ["CPUExecutionProvider"]
)



# ANALYTICAL FEATURES

#  Tripwire (footfall counter) 
# A horizontal line drawn across the frame.  Every person whose bounding-box centre crosses this Y coordinate is counted as an entry or exit.

# For a 480p (640×480) frame, y = 240 places the line at the vertical midpoint.
TRIPWIRE_Y = 240  # pixels from the top of the frame

# Minimum number of frames a track must sustain crossing momentum before the crossing is registered (prevents jitter-induced double-counting).
TRIPWIRE_DEBOUNCE_FRAMES = 3

#  Dwell-time zones 
# Each zone is a named bounding rectangle: (x1, y1, x2, y2) in pixel coords relative to the ORIGINAL video frame (before any model resize).
# Add, rename, or remove zones here without touching any other file.
DWELL_ZONES = {
    "billing_counter": (480, 300, 620, 460),   # bottom-right quadrant (example)
    # "entrance":       (  0,   0, 160, 200),  # uncomment to add more zones
}

# Minimum continuous seconds a person must stay inside a zone to be logged.
DWELL_MIN_SECONDS = 2.0



# VISUALISATION

# Heatmap colour map applied when overlaying the accumulation array on video.
# Any OpenCV / Matplotlib colourmap name is valid (e.g. "JET", "HOT", "INFERNO").
HEATMAP_COLORMAP = "JET"

# Alpha blending weight for the heatmap overlay [0.0 = invisible, 1.0 = opaque].
HEATMAP_ALPHA = 0.4

# Bounding-box colour for detected persons (BGR format for OpenCV).
BBOX_COLOR = (0, 255, 0)       # green
BBOX_THICKNESS = 2

# Tripwire line colour and thickness.
TRIPWIRE_COLOR = (0, 0, 255)   # red
TRIPWIRE_THICKNESS = 2

# Dwell-zone rectangle colour and thickness.
DWELL_ZONE_COLOR = (255, 165, 0)  # orange
DWELL_ZONE_THICKNESS = 2


# TRAINING (used by training.py)

TRAIN_EPOCHS   = 50
TRAIN_BATCH    = 8
TRAIN_IMG_SIZE = 640                                            # square side in px
TRAIN_DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
TRAIN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "train")