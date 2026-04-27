import os
import time
import shutil
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

'''
# Import config as the single source of truth

from config import (
    DATASET_YAML,
    EPOCHS,
    IMG_SIZE,
    BATCH_SIZE,
    DEVICE,
    MODELS_DIR,
    RUNS_DIR,
    YOLOV8N_ONNX,
    YOLOV5N_ONNX,
)'''

# Import config as the single source of truth
from config import (
    TRAIN_DATA_YAML,
    TRAIN_EPOCHS,
    TRAIN_IMG_SIZE,
    TRAIN_BATCH,
    USE_CUDA,
    PROJECT_ROOT,
    MODEL_YOLOV8N_ONNX,
    MODEL_YOLOV5N_ONNX,
)

# Mapping config variables to what the training script expects
DATASET_YAML = TRAIN_DATA_YAML
EPOCHS       = TRAIN_EPOCHS
IMG_SIZE     = TRAIN_IMG_SIZE
BATCH_SIZE   = TRAIN_BATCH
DEVICE       = 0 if USE_CUDA else "cpu"  # 0 targets your GTX 1650 Ti
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RUNS_DIR     = os.path.join(PROJECT_ROOT, "runs")
YOLOV8N_ONNX = MODEL_YOLOV8N_ONNX
YOLOV5N_ONNX = MODEL_YOLOV5N_ONNX



# Helper utilities

def ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    for directory in [MODELS_DIR, RUNS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directories ready: '{MODELS_DIR}/', '{RUNS_DIR}/'")


def _load_results_csv(run_dir: str) -> pd.DataFrame:
    """
    Load the Ultralytics results.csv produced after training.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory (e.g. runs/detect/train/).

    Returns
    -------
    pd.DataFrame
        DataFrame with stripped column names, or an empty DataFrame on error.
    """
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        print(f"[WARNING] results.csv not found at '{csv_path}'")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()   # Ultralytics adds leading spaces
    return df


def _save_loss_graph(df: pd.DataFrame, model_name: str) -> None:
    """
    Plot and save the training loss curves (box loss + class loss) vs epoch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from results.csv.
    model_name : str
        Short model identifier used in the filename (e.g. 'yolov8n').
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Column names differ slightly between YOLOv8 and YOLOv5 — handle both
    box_col  = next((c for c in df.columns if "box_loss" in c.lower()), None)
    cls_col  = next((c for c in df.columns if "cls_loss" in c.lower()), None)
    epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)

    x = df[epoch_col] if epoch_col else range(len(df))

    if box_col:
        ax.plot(x, df[box_col], label="Box Loss",   color="#E63946", linewidth=2)
    if cls_col:
        ax.plot(x, df[cls_col], label="Class Loss", color="#457B9D", linewidth=2)

    ax.set_title(f"{model_name.upper()} — Training Loss vs Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out_path = Path(RUNS_DIR) / f"{model_name}_loss.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss graph saved → '{out_path}'")


def _save_map_graph(df: pd.DataFrame, model_name: str) -> None:
    """
    Plot and save the mAP50 curve vs epoch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from results.csv.
    model_name : str
        Short model identifier used in the filename (e.g. 'yolov8n').
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    map_col   = next((c for c in df.columns if "map50" in c.lower() and "map50-95" not in c.lower()), None)
    epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)

    x = df[epoch_col] if epoch_col else range(len(df))

    if map_col:
        ax.plot(x, df[map_col], label="mAP@50", color="#2A9D8F", linewidth=2)
    else:
        print(f"[WARNING] mAP50 column not found for {model_name}; graph will be empty.")

    ax.set_title(f"{model_name.upper()} — mAP50 vs Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.50")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out_path = Path(RUNS_DIR) / f"{model_name}_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] mAP graph  saved → '{out_path}'")


def _best_map50(df: pd.DataFrame) -> float:
    """
    Return the highest mAP50 value recorded across all epochs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from results.csv.

    Returns
    -------
    float
        Best mAP50 value, or 0.0 if unavailable.
    """
    map_col = next((c for c in df.columns if "map50" in c.lower() and "map50-95" not in c.lower()), None)
    if map_col and not df.empty:
        return float(df[map_col].max())
    return 0.0


def _copy_best_weights(run_dir: str, dest_onnx: str, model_name: str) -> str:
    """
    Locate best.pt inside the Ultralytics run directory and export it to ONNX.

    Parameters
    ----------
    run_dir : str
        Path to the Ultralytics training run directory.
    dest_onnx : str
        Destination path for the exported .onnx file.
    model_name : str
        Human-readable model name for log messages.

    Returns
    -------
    str
        Absolute path to the exported ONNX file, or empty string on failure.
    """
    best_pt = Path(run_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] best.pt not found at '{best_pt}' — skipping ONNX export for {model_name}.")
        return ""

    print(f"[INFO] Exporting {model_name} best.pt → ONNX ...")
    try:
        # Use Ultralytics export API — works identically for v8 and v5
        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        # Export returns path to the generated .onnx next to best.pt
        exported = model.export(format="onnx", imgsz=IMG_SIZE, dynamic=False, simplify=True)
        exported_path = Path(exported) if exported else best_pt.with_suffix(".onnx")
        # Move to models/ with the canonical project name
        shutil.move(str(exported_path), dest_onnx)
        print(f"[INFO] ONNX model saved → '{dest_onnx}'")
        return dest_onnx
    except Exception as exc:
        print(f"[ERROR] ONNX export failed for {model_name}: {exc}")
        return ""



# Per-model training functions

def train_yolov8n() -> dict:
    """
    Train YOLOv8n using Ultralytics YOLO API.

    Reads all hyperparameters from config.py.  After training the function:
      1. Saves loss and mAP graphs to RUNS_DIR.
      2. Exports best.pt to ONNX and copies it to MODELS_DIR/yolov8n.onnx.

    Returns
    -------
    dict
        Summary dict with keys: model, best_map50, train_time_min, onnx_path.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  Training  :  YOLOv8n")
    print("=" * 60)

    model = YOLO("yolov8n.pt")   # downloads pretrained weights if absent

    t0 = time.time()
    results = model.train(
        data    = DATASET_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = "RUNS_DIR",
        name    = "yolov8n_train",
        exist_ok= True,
        verbose = False,
    )
    elapsed = (time.time() - t0) / 60.0

    run_dir = str(Path("runs/detect/yolov8n_train"))
    df      = _load_results_csv(run_dir)
    _save_loss_graph(df, "yolov8n")
    _save_map_graph(df,  "yolov8n")

    best_map = _best_map50(df)
    onnx_out = _copy_best_weights(run_dir, YOLOV8N_ONNX, "YOLOv8n")

    summary = {
        "model"          : "YOLOv8n",
        "best_map50"     : best_map,
        "train_time_min" : elapsed,
        "onnx_path"      : onnx_out,
    }
    print(f"[INFO] YOLOv8n training complete — {elapsed:.1f} min | best mAP50: {best_map:.4f}")
    return summary


def train_yolov5n() -> dict:
    """
    Train YOLOv5n using the Ultralytics YOLO API (YOLOv5 is supported via
    'yolov5nu' weights in modern Ultralytics builds).

    Reads all hyperparameters from config.py.  After training the function:
      1. Saves loss and mAP graphs to RUNS_DIR.
      2. Exports best.pt to ONNX and copies it to MODELS_DIR/yolov5n.onnx.

    Returns
    -------
    dict
        Summary dict with keys: model, best_map50, train_time_min, onnx_path.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  Training  :  YOLOv5n")
    print("=" * 60)

    # 'yolov5nu.pt' = YOLOv5n with Ultralytics v8-compatible head
    model = YOLO("yolov5nu.pt")

    t0 = time.time()
    results = model.train(
        data    = DATASET_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = "RUNS_DIR",
        name    = "yolov5n_train",
        exist_ok= True,
        verbose = False,
        workers = 2,
    )
    elapsed = (time.time() - t0) / 60.0

    run_dir = str(Path("runs/detect/yolov5n_train"))
    df      = _load_results_csv(run_dir)
    _save_loss_graph(df, "yolov5n")
    _save_map_graph(df,  "yolov5n")

    best_map = _best_map50(df)
    onnx_out = _copy_best_weights(run_dir, YOLOV5N_ONNX, "YOLOv5n")

    summary = {
        "model"          : "YOLOv5n",
        "best_map50"     : best_map,
        "train_time_min" : elapsed,
        "onnx_path"      : onnx_out,
    }
    print(f"[INFO] YOLOv5n training complete — {elapsed:.1f} min | best mAP50: {best_map:.4f}")
    return summary



# Summary printer

def print_summary(summaries: list) -> None:
    """
    Print a formatted summary table of all trained models.

    Parameters
    ----------
    summaries : list of dict
        Each dict must contain: model, best_map50, train_time_min, onnx_path.
    """
    print("\n" + "=" * 70)
    print(f"{'MODEL':<15} {'BEST mAP50':>12} {'TRAIN TIME (min)':>18} {'ONNX EXPORTED':>15}")
    print("-" * 70)
    for s in summaries:
        exported = "YES" if s.get("onnx_path") else "FAILED"
        print(
            f"{s['model']:<15} "
            f"{s['best_map50']:>12.4f} "
            f"{s['train_time_min']:>18.1f} "
            f"{exported:>15}"
        )
    print("=" * 70)
    print("Note: SSD-MobileNet V2 is benchmarked separately in benchmark.py")
    print()



# Entry point

def main() -> None:
    """
    Orchestrate full ERVS training pipeline:
      1. Prepare output directories.
      2. Train YOLOv8n  → graphs + ONNX export.
      3. Train YOLOv5n  → graphs + ONNX export.
      4. Print final summary table.
    """
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ERVS — Model Training Pipeline            ║")
    print("║   Models : YOLOv8n | YOLOv5n               ║")
    print("╚══════════════════════════════════════════════╝\n")

    ensure_dirs()

    summaries = []
    summaries.append(train_yolov8n())
    summaries.append(train_yolov5n())

    print_summary(summaries)
    print("[INFO] All training complete. Graphs saved to runs/  |  ONNX models saved to models/")


if __name__ == "__main__":
    main()