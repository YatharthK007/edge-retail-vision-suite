import os
import csv
import math
import shutil
import random
from pathlib import Path

import cv2


#  CONFIG 

VIDEO_PATH       = "dataset/TownCentreXVID.mp4"          # path to the raw video file
ANNOTATIONS_PATH = "dataset/TownCentre_groundtruth.csv"   # path to the annotation CSV
OUTPUT_DIR       = "dataset"                       # root of the YOLO dataset
FRAME_STEP       = 5          # extract every N-th frame (5 → ~20 % of frames)
VAL_SPLIT        = 0.20       # fraction of frames reserved for validation
RANDOM_SEED      = 42         # for reproducible train/val split

# YOLO class definition — Oxford Town Centre has only one object class.
CLASS_NAMES = ["person"]
PERSON_CLASS_ID = 0


#  CSV COLUMN INDICES
# Column layout (0-based):
#   0  personNumber
#   1  frameNumber
#   2  lost        (1 = person is off-screen / lost, skip these)
#   3  occluded    (informational, we keep occluded persons)
#   4  generated   (informational)
#   5  label       (always "Pedestrian")
#   6  x1  (left,   pixel coords in original video resolution)
#   7  y1  (top)
#   8  x2  (right)
#   9  y2  (bottom)

'''COL_FRAME   = 1
COL_LOST    = 2
COL_X1      = 6
COL_Y1      = 7
COL_X2      = 8
COL_Y2      = 9'''

COL_FRAME   = 1
COL_VALID   = 3  # bodyValid (1 = valid person, 0 = invalid/off-screen)
COL_X1      = 8  # bodyLeft
COL_Y1      = 9  # bodyTop
COL_X2      = 10 # bodyRight
COL_Y2      = 11 # bodyBottom


#  HELPERS

def make_dirs(base: str) -> dict:
    """
    Create the full YOLO directory tree under *base* and return a dict of paths.

    Parameters
    ----------
    base : str
        Root output directory (e.g. "dataset").

    Returns
    -------
    dict
        Keys: "img_train", "img_val", "lbl_train", "lbl_val"
    """
    paths = {
        "img_train": os.path.join(base, "images", "train"),
        "img_val":   os.path.join(base, "images", "val"),
        "lbl_train": os.path.join(base, "labels", "train"),
        "lbl_val":   os.path.join(base, "labels", "val"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def write_yaml(base: str, class_names: list) -> None:
    """
    Write the data.yaml file that Ultralytics training expects.

    Parameters
    ----------
    base : str
        Root dataset directory.
    class_names : list of str
        Ordered list of class names.
    """
    abs_base = str(Path(base).resolve())
    yaml_content = (
        f"path: {abs_base}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = os.path.join(base, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(yaml_content)
    print(f"[INFO] data.yaml written → '{yaml_path}'")


def load_annotations(csv_path: str) -> dict:
    """
    Parse the Oxford Town Centre CSV into a frame-indexed dict.

    Skips rows where the person is marked as 'lost' (off-screen).
    Handles both header and header-less CSV variants automatically.

    Parameters
    ----------
    csv_path : str
        Path to TownCentre_groundtruth.csv.

    Returns
    -------
    dict
        {frame_number (int): [(x1, y1, x2, y2), ...]}
        Pixel coordinates in the original video resolution.
    """
    annotations: dict = {}

    with open(csv_path, newline="") as fh:
        sample = fh.read(1024)
        fh.seek(0)
        # Detect whether the first row is a header (contains non-numeric text)
        has_header = not sample.split("\n")[0].split(",")[0].strip().lstrip("-").isdigit()
        reader = csv.reader(fh)
        if has_header:
            next(reader)   # skip header row

        for row in reader:
            if len(row) < 12:
                continue   # malformed / empty row

            try:
                frame_num  = int(float(row[COL_FRAME]))
                body_valid = int(float(row[COL_VALID]))
                x1 = float(row[COL_X1])
                y1 = float(row[COL_Y1])
                x2 = float(row[COL_X2])
                y2 = float(row[COL_Y2])
            except ValueError:
                continue   # header row sneaked through or corrupt data
                
            if body_valid == 0:
                continue   # person is off-screen — skip
                
            # Ensure x1 < x2 and y1 < y2 (some rows have swapped coords)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if x2 <= x1 or y2 <= y1:
                continue   # degenerate box

            annotations.setdefault(frame_num, []).append((x1, y1, x2, y2))

    total_boxes = sum(len(v) for v in annotations.values())
    print(f"[INFO] Loaded annotations: {len(annotations)} frames with {total_boxes} person boxes.")
    return annotations


def to_yolo_line(x1: float, y1: float, x2: float, y2: float,
                 frame_w: int, frame_h: int, class_id: int) -> str:
    """
    Convert a pixel bounding box to a normalised YOLO annotation string.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalised to [0, 1] by the frame dimensions.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Pixel coordinates of the bounding box corners.
    frame_w, frame_h : int
        Width and height of the video frame.
    class_id : int
        Integer class index (0 for "person").

    Returns
    -------
    str
        Single YOLO annotation line (no trailing newline).
    """
    x_center = ((x1 + x2) / 2.0) / frame_w
    y_center = ((y1 + y2) / 2.0) / frame_h
    width    = (x2 - x1) / frame_w
    height   = (y2 - y1) / frame_h

    # Clamp to [0, 1] — handles any minor out-of-frame annotations
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"




#  MAIN 

def main() -> None:
    """
    Full pipeline:
      1. Validate input files exist.
      2. Create output directory structure.
      3. Extract every FRAME_STEP-th frame that has at least one annotation.
      4. Write the corresponding YOLO .txt label file per frame.
      5. Split frames into train / val sets (80 / 20).
      6. Write data.yaml.
      7. Print final summary.
    """

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   ERVS — Oxford Town Centre Dataset Preparation     ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    #  1. Validate inputs 
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: '{VIDEO_PATH}'\n"
            f"Place TownCentreXVID.mp4 in the same folder as this script, "
            f"or update VIDEO_PATH in the CONFIG block."
        )
    if not os.path.isfile(ANNOTATIONS_PATH):
        raise FileNotFoundError(
            f"Annotations not found: '{ANNOTATIONS_PATH}'\n"
            f"Place TownCentre_groundtruth.csv in the same folder, "
            f"or update ANNOTATIONS_PATH in the CONFIG block."
        )

    #  2. Create directories
    dirs = make_dirs(OUTPUT_DIR)

    #  3. Load annotations 
    annotations = load_annotations(ANNOTATIONS_PATH)

    #  4. Open video and extract frames 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: '{VIDEO_PATH}'")

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video : {frame_w}×{frame_h} px | {total_frames} total frames")
    print(f"[INFO] Extracting every {FRAME_STEP}th frame that has annotations …\n")

    # Collect (frame_index, image_array, boxes) for frames that pass the filter
    kept_frames: list = []   # list of (frame_idx, img, boxes)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0 and frame_idx in annotations:
            boxes = annotations[frame_idx]
            kept_frames.append((frame_idx, frame.copy(), boxes))

        frame_idx += 1

    cap.release()
    print(f"[INFO] Kept {len(kept_frames)} frames (every {FRAME_STEP}th with ≥1 annotation).")

    if not kept_frames:
        print("[ERROR] No frames matched any annotation. "
              "Check that the CSV frame numbers align with the video.")
        return

    #  5. Train / val split 
    random.seed(RANDOM_SEED)
    random.shuffle(kept_frames)

    n_val   = max(1, math.floor(len(kept_frames) * VAL_SPLIT))
    n_train = len(kept_frames) - n_val

    train_frames = kept_frames[n_val:]    # 80 %
    val_frames   = kept_frames[:n_val]    # 20 %

    #  6. Write images + labels 
    def save_split(frames: list, img_dir: str, lbl_dir: str, split_name: str) -> int:
        """
        Save image files and corresponding YOLO label text files for one split.

        Parameters
        ----------
        frames : list of (frame_idx, img, boxes)
            Frames belonging to this split.
        img_dir : str
            Destination directory for .jpg images.
        lbl_dir : str
            Destination directory for .txt label files.
        split_name : str
            Human-readable split name ("train" or "val") for log messages.

        Returns
        -------
        int
            Number of images written.
        """
        count = 0
        for frame_idx, img, boxes in frames:
            stem = f"frame_{frame_idx:06d}"
            img_path = os.path.join(img_dir, stem + ".jpg")
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            # Save image
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write YOLO label file (one box per line)
            lines = [
                to_yolo_line(x1, y1, x2, y2, frame_w, frame_h, PERSON_CLASS_ID)
                for x1, y1, x2, y2 in boxes
            ]
            with open(lbl_path, "w") as fh:
                fh.write("\n".join(lines) + "\n")

            count += 1
            if count % 200 == 0:
                print(f"  [{split_name}] Written {count}/{len(frames)} …")

        return count

    print(f"\n[INFO] Writing train split ({n_train} frames) …")
    n_written_train = save_split(train_frames, dirs["img_train"], dirs["lbl_train"], "train")

    print(f"[INFO] Writing val split   ({n_val} frames) …")
    n_written_val = save_split(val_frames, dirs["img_val"], dirs["lbl_val"], "val")

    #  7. Write data.yaml
    write_yaml(OUTPUT_DIR, CLASS_NAMES)

    #  8. Summary
    print("\n" + "=" * 56)
    print(f"  Dataset preparation complete!")
    print("=" * 56)
    print(f"  Train images : {n_written_train}")
    print(f"  Val   images : {n_written_val}")
    print(f"  Total        : {n_written_train + n_written_val}")
    print(f"  Frame size   : {frame_w} × {frame_h} px (original)")
    print(f"  Output dir   : {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 56)
    print("\n  Next step → run:  python training.py\n")


if __name__ == "__main__":
    main()