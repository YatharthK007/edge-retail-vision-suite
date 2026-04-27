import os
import time
import urllib.request

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend 
import matplotlib.pyplot as plt

# Re-use our own inference pipeline for the YOLO models (rule: no duplication)
from inference import load_model, run_inference
from config import (
    PROJECT_ROOT,
    INPUT_SOURCE,
    MODEL_YOLOV8N_ONNX,
    MODEL_YOLOV5N_ONNX,
    CONFIDENCE_THRESHOLD,
    INPUT_SIZE,
)



# Constants

BENCHMARK_FRAMES = 100          # number of frames evaluated per model

# Hardcoded mAP values from training runs (professor's requirement)
MAP_SCORES = {
    "YOLOv8n":       98.6,
    "YOLOv5n":       98.7,
    "SSD-MobileNet": 72.7,
}

# SSD-MobileNet V2 Caffe model URLs
SSD_PROTOTXT_URL   = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
SSD_CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/mobilenet_iter_73000.caffemodel"
)

# Local paths for SSD assets (kept inside models/ like everything else)
SSD_PROTOTXT_PATH    = os.path.join(PROJECT_ROOT, "models", "deploy.prototxt")
SSD_CAFFEMODEL_PATH  = os.path.join(
    PROJECT_ROOT, "models", "mobilenet_iter_73000.caffemodel"
)

# Output directory for all benchmark artefacts
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
RESULTS_TXT  = os.path.join(RUNS_DIR, "benchmark_results.txt")
RESULTS_CHART = os.path.join(RUNS_DIR, "benchmark_chart.png")

# COCO person class index used by the SSD Caffe model
SSD_PERSON_CLASS_ID = 15        # MobileNet-SSD PASCAL VOC label 15 = person



# Helper: ensure output directories exist

def _ensure_dirs() -> None:
    """Create models/ and runs/ directories if they do not already exist."""
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)



# SSD-MobileNet: download weights if missing

def download_ssd_weights() -> None:
    """
    Download the SSD-MobileNet V2 Caffe prototxt and caffemodel files into
    the models/ directory if they are not already present.

    Uses urllib (stdlib only — no extra dependency) and prints progress so
    the user knows what is happening during first run.

    Raises:
        urllib.error.URLError: If the remote files cannot be reached.
    """
    assets = [
        (SSD_PROTOTXT_URL,    SSD_PROTOTXT_PATH,   "deploy.prototxt"),
        (SSD_CAFFEMODEL_URL,  SSD_CAFFEMODEL_PATH, "mobilenet_iter_73000.caffemodel"),
    ]
    for url, dest, label in assets:
        if os.path.exists(dest):
            print(f"[benchmark] Found  : {label}  (skipping download)")
        else:
            print(f"[benchmark] Downloading {label} …")
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[benchmark] Saved  : {label}  ({size_mb:.1f} MB)")



# SSD-MobileNet: load Caffe network

def load_ssd_model() -> cv2.dnn_Net:
    """
    Load the SSD-MobileNet V2 Caffe model using OpenCV's DNN module.

    Returns:
        cv2.dnn_Net: Loaded network ready for forward passes.

    Raises:
        cv2.error: If the prototxt or caffemodel files are invalid.
    """
    print(f"[benchmark] Loading SSD-MobileNet from Caffe weights …")
    net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT_PATH, SSD_CAFFEMODEL_PATH)
    print("[benchmark] SSD-MobileNet loaded successfully.\n")
    return net



# SSD-MobileNet: single-frame inference

def run_ssd_inference(
    net: cv2.dnn_Net,
    frame: np.ndarray,
) -> tuple[list, list, float, float]:
    """
    Run SSD-MobileNet V2 inference on a single raw BGR frame.

    The Caffe model expects a 300×300 blob with pixel-mean subtraction
    (mean = [127.5, 127.5, 127.5]) and a scale factor of 1/127.5.

    Only detections whose class label matches SSD_PERSON_CLASS_ID and whose
    confidence exceeds CONFIDENCE_THRESHOLD (from config.py) are kept.

    Args:
        net   (cv2.dnn_Net) : Loaded SSD Caffe network.
        frame (np.ndarray)  : Raw BGR frame as read by cv2.VideoCapture.

    Returns:
        tuple:
            boxes            (list[list[int]]): Detected boxes as
                             [x1, y1, x2, y2] in original frame pixel space.
            confidences      (list[float])    : Confidence per box.
            inference_time_ms (float)         : Wall-clock forward-pass time.
            fps               (float)         : 1000 / inference_time_ms.
    """
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)

    t_start = time.perf_counter()
    detections = net.forward()          # shape: (1, 1, N, 7)
    t_end = time.perf_counter()

    inference_time_ms = (t_end - t_start) * 1000.0
    fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

    boxes: list[list[int]] = []
    confidences: list[float] = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id   = int(detections[0, 0, i, 1])

        if class_id != SSD_PERSON_CLASS_ID:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Detections are normalised [0, 1] — scale back to frame pixels
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        boxes.append([x1, y1, x2, y2])
        confidences.append(round(confidence, 4))

    return boxes, confidences, inference_time_ms, fps



# Core benchmark loop

def _prep_yolo_tensor(frame: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR frame into the float32 CHW tensor expected by
    run_inference() from inference.py.

    Mirrors what preprocessing.py does so benchmark.py is self-contained
    and does not depend on preprocessing.py being present yet.

    Args:
        frame (np.ndarray): Raw BGR frame (any resolution).

    Returns:
        np.ndarray: Float32 array of shape (1, 3, 640, 640), values in [0, 1].
    """
    resized = cv2.resize(frame, INPUT_SIZE)                 # (640, 640, 3)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      # BGR → RGB
    chw     = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)                      # (1, 3, 640, 640)


def benchmark_model(
    model_name: str,
    session_or_net,
    cap: cv2.VideoCapture,
    num_frames: int = BENCHMARK_FRAMES,
) -> dict:
    """
    Run one model against the first `num_frames` frames of an already-opened
    VideoCapture and return timing statistics.

    For YOLOv8n / YOLOv5n the session_or_net is an ort.InferenceSession and
    run_inference() from inference.py is called.  For SSD-MobileNet it is a
    cv2.dnn_Net and run_ssd_inference() is called.

    Args:
        model_name     (str)                       : Human-readable model name.
        session_or_net (ort.InferenceSession | Net): Loaded model object.
        cap            (cv2.VideoCapture)           : Open video capture,
                                                     rewound to frame 0.
        num_frames     (int)                        : Frames to evaluate.

    Returns:
        dict with keys:
            "model"        – model_name
            "avg_fps"      – float
            "avg_ms"       – float
            "map"          – float  (hardcoded from MAP_SCORES)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # always start from frame 0

    times_ms: list[float] = []
    frames_processed = 0

    print(f"[benchmark] ── {model_name} ──────────────────────────────────")

    while frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"[benchmark] Video ended at frame {frames_processed}.")
            break

        if model_name in ("YOLOv8n", "YOLOv5n"):
            tensor = _prep_yolo_tensor(frame)
            _, _, inf_ms, _ = run_inference(session_or_net, tensor)
        else:
            # SSD-MobileNet — pass raw frame; resizing is done inside
            _, _, inf_ms, _ = run_ssd_inference(session_or_net, frame)

        times_ms.append(inf_ms)
        frames_processed += 1

        if frames_processed % 25 == 0:
            running_avg = sum(times_ms) / len(times_ms)
            print(
                f"[benchmark]   frame {frames_processed:>3}/{num_frames}"
                f"  |  this frame: {inf_ms:6.2f} ms"
                f"  |  running avg: {running_avg:6.2f} ms"
            )

    if not times_ms:
        return {"model": model_name, "avg_fps": 0.0, "avg_ms": 0.0,
                "map": MAP_SCORES[model_name]}

    avg_ms  = sum(times_ms) / len(times_ms)
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(
        f"[benchmark]   DONE  |  avg {avg_ms:.2f} ms  |  avg {avg_fps:.1f} FPS\n"
    )

    return {
        "model":   model_name,
        "avg_fps": round(avg_fps, 2),
        "avg_ms":  round(avg_ms,  2),
        "map":     MAP_SCORES[model_name],
    }



# Winner selection

def select_winner(results: list[dict]) -> str:
    """
    Apply the project's winner-selection rules to the benchmark results.

    Rules (in priority order):
        1. YOLOv8n  wins if its mAP > 80 % AND avg FPS > 15.
        2. YOLOv5n  wins if its avg FPS > 18.
        3. SSD-MobileNet wins otherwise.

    Args:
        results (list[dict]): List of result dicts from benchmark_model().

    Returns:
        str: Name of the winning model.
    """
    # Build a quick lookup by model name
    lookup = {r["model"]: r for r in results}

    v8  = lookup.get("YOLOv8n",       {"avg_fps": 0.0, "map": 0.0})
    v5  = lookup.get("YOLOv5n",       {"avg_fps": 0.0, "map": 0.0})

    if v8["map"] > 80.0 and v8["avg_fps"] > 15.0:
        return "YOLOv8n"
    elif v5["avg_fps"] > 18.0:
        return "YOLOv5n"
    else:
        return "SSD-MobileNet"



# Output: console table + .txt file

def print_and_save_table(results: list[dict], winner: str) -> None:
    """
    Print a formatted comparison table to stdout and write the same text to
    runs/benchmark_results.txt.

    Table columns: Model | Avg FPS | Avg Inference (ms) | mAP (%)

    Args:
        results (list[dict]): Benchmark result dicts.
        winner  (str)        : Name of the winning model.
    """
    sep   = "+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 22 + "+" + "-" * 10 + "+"
    header = (
        f"| {'Model':<16} | {'Avg FPS':>10} | {'Avg Inference (ms)':>20} | "
        f"{'mAP (%)':>8} |"
    )

    lines = [
        "",
        "=" * 66,
        "  ERVS — Model Benchmark Results",
        "=" * 66,
        sep,
        header,
        sep,
    ]

    for r in results:
        tag = "  ◀ WINNER" if r["model"] == winner else ""
        row = (
            f"| {r['model']:<16} | {r['avg_fps']:>10.1f} | "
            f"{r['avg_ms']:>20.2f} | {r['map']:>7.1f}% |"
            f"{tag}"
        )
        lines.append(row)

    lines += [
        sep,
        "",
        f"  WINNER: {winner} — Best balance of FPS and accuracy",
        "",
        "=" * 66,
        "",
    ]

    output = "\n".join(lines)
    print(output)

    with open(RESULTS_TXT, "w", encoding="utf-8") as fh:
        fh.write(output)
    print(f"[benchmark] Table saved → {RESULTS_TXT}")



# Output: bar chart

def save_chart(results: list[dict], winner: str) -> None:
    """
    Generate and save a side-by-side bar chart comparing Avg FPS and mAP (%)
    across all three models.

    Two sub-plots share the same x-axis (model names):
        - Left  : Avg FPS  (blue bars)
        - Right : mAP (%)  (orange bars)

    The winning model's bars are highlighted with a bold edge.

    Args:
        results (list[dict]): Benchmark result dicts.
        winner  (str)        : Name of the winning model — its bars receive
                               a thick black edge for visual emphasis.
    """
    model_names = [r["model"]   for r in results]
    fps_values  = [r["avg_fps"] for r in results]
    map_values  = [r["map"]     for r in results]

    # Edge widths: thicker for the winner column
    fps_edges = [3.0 if n == winner else 0.8 for n in model_names]
    map_edges = [3.0 if n == winner else 0.8 for n in model_names]

    x = np.arange(len(model_names))
    bar_w = 0.35

    fig, (ax_fps, ax_map) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    fig.suptitle(
        "ERVS — Model Benchmark Comparison", fontsize=14, fontweight="bold"
    )

    #  FPS subplot 
    bars_fps = ax_fps.bar(
        x, fps_values,
        width=bar_w * 2,
        color="#4C72B0",
        edgecolor="black",
        linewidth=fps_edges,
        zorder=3,
    )
    ax_fps.set_title("Average FPS (higher = better)", fontsize=11)
    ax_fps.set_ylabel("Frames per Second")
    ax_fps.set_xticks(x)
    ax_fps.set_xticklabels(model_names, fontsize=10)
    ax_fps.set_ylim(0, max(fps_values) * 1.25 if fps_values else 1)
    ax_fps.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax_fps.set_axisbelow(True)

    for bar, val in zip(bars_fps, fps_values):
        ax_fps.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(fps_values) * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    #  mAP subplot
    bars_map = ax_map.bar(
        x, map_values,
        width=bar_w * 2,
        color="#DD8452",
        edgecolor="black",
        linewidth=map_edges,
        zorder=3,
    )
    ax_map.set_title("mAP — % (higher = better)", fontsize=11)
    ax_map.set_ylabel("mAP (%)")
    ax_map.set_xticks(x)
    ax_map.set_xticklabels(model_names, fontsize=10)
    ax_map.set_ylim(0, 115)
    ax_map.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax_map.set_axisbelow(True)

    for bar, val in zip(bars_map, map_values):
        ax_map.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Winner annotation on both subplots
    winner_idx = model_names.index(winner)
    for ax in (ax_fps, ax_map):
        ax.get_xticklabels()[winner_idx].set_color("#B00020")
        ax.get_xticklabels()[winner_idx].set_fontweight("bold")

    fig.text(
        0.5, -0.02,
        f"★  WINNER: {winner} — Best balance of FPS and accuracy  ★",
        ha="center", fontsize=11, color="#B00020", fontweight="bold",
    )

    plt.savefig(RESULTS_CHART, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Chart saved  → {RESULTS_CHART}")



# Entry point

def main() -> None:
    """
    Orchestrate the full benchmark pipeline:

        1. Ensure output directories exist.
        2. Download SSD-MobileNet weights if missing.
        3. Load all three models.
        4. Open the input video from config.INPUT_SOURCE.
        5. Run each model over the first BENCHMARK_FRAMES frames.
        6. Select the winner.
        7. Print + save the comparison table.
        8. Generate + save the bar chart.
        9. Print the final winner announcement.
    """
    _ensure_dirs()

    #  Step 1: SSD weights
    download_ssd_weights()

    #  Step 2: Load models 
    print("\n[benchmark] Loading models …\n")

    yolov8n_session  = load_model(MODEL_YOLOV8N_ONNX)
    yolov5n_session  = load_model(MODEL_YOLOV5N_ONNX)
    ssd_net          = load_ssd_model()

    #  Step 3: Open video
    print(f"[benchmark] Opening video: {INPUT_SOURCE}\n")
    cap = cv2.VideoCapture(INPUT_SOURCE)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"[benchmark] Cannot open video: {INPUT_SOURCE}\n"
            f"  Check INPUT_SOURCE in config.py."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[benchmark] Video info: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  |  "
        f"{total_frames} total frames  |  "
        f"Benchmarking first {BENCHMARK_FRAMES} frames per model.\n"
    )

    if total_frames < BENCHMARK_FRAMES:
        print(
            f"[benchmark] WARNING: video has only {total_frames} frames — "
            f"benchmark will use all available frames."
        )

    #  Step 4: Run benchmarks 
    results = []
    results.append(benchmark_model("YOLOv8n",       yolov8n_session, cap))
    results.append(benchmark_model("YOLOv5n",       yolov5n_session, cap))
    results.append(benchmark_model("SSD-MobileNet", ssd_net,         cap))

    cap.release()

    #  Step 5: Select winner 
    winner = select_winner(results)

    # Step 6: Print + save table 
    print_and_save_table(results, winner)

    #  Step 7: Save chart 
    save_chart(results, winner)

    # Step 8: Final announcement 
    print(f"\n  ★  WINNER: {winner} — Best balance of FPS and accuracy  ★\n")


if __name__ == "__main__":
    main()