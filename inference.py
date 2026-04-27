import time
import numpy as np
import cv2
import onnxruntime as ort

from config import (
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    ONNX_PROVIDERS,
    INPUT_SIZE,
)



# Model Loading

def load_model(model_path: str) -> ort.InferenceSession:
    """
    Load an ONNX model from disk and create an InferenceSession.

    Uses ONNX_PROVIDERS from config.py so the same code works on the
    laptop (CUDAExecutionProvider) and on the Jetson Nano
    (CPUExecutionProvider) without any modification.

    Args:
        model_path (str): Absolute or relative path to the .onnx file.

    Returns:
        ort.InferenceSession: A ready-to-use ONNX Runtime inference session.

    Raises:
        FileNotFoundError: If the model file does not exist at model_path.
        RuntimeError: If ONNX Runtime cannot load the model.
    """
    print(f"[inference] Loading model from: {model_path}")
    print(f"[inference] Using providers   : {ONNX_PROVIDERS}")

    session = ort.InferenceSession(model_path, providers=ONNX_PROVIDERS)

    # Print input metadata
    print("[inference] Model inputs:")
    for inp in session.get_inputs():
        print(f"   └─ name='{inp.name}'  shape={inp.shape}  dtype={inp.type}")

    # Print output metadata
    print("[inference] Model outputs:")
    for out in session.get_outputs():
        print(f"   └─ name='{out.name}'  shape={out.shape}  dtype={out.type}")

    print("[inference] Model loaded successfully.\n")
    return session



# Core Inference

def run_inference(
    session: ort.InferenceSession,
    preprocessed_frame: np.ndarray,
) -> tuple[list, list, float, float]:
    """
    Run person detection on a single preprocessed frame.

    Handles the YOLOv8n ONNX output tensor of shape [1, 5, 8400]:
        - Axis 1, rows 0-3 : bounding box as [cx, cy, w, h] (normalised to
          INPUT_SIZE space — i.e. pixel values in the 640×640 frame).
        - Axis 1, row  4   : person-class confidence score.

    Steps:
        1. Feed the frame through the ONNX session and time the call.
        2. Transpose the raw output to [8400, 5] for easy row-wise access.
        3. Filter anchors whose confidence ≥ CONFIDENCE_THRESHOLD.
        4. Convert [cx, cy, w, h] → [x1, y1, w, h] for cv2.dnn.NMSBoxes.
        5. Apply NMS and collect surviving boxes as [x1, y1, x2, y2].
        6. Return detections along with inference_time_ms and fps.

    Args:
        session (ort.InferenceSession): A loaded ONNX inference session
            (returned by load_model).
        preprocessed_frame (np.ndarray): Float32 numpy array of shape
            (1, 3, 640, 640) — the CHW-batched tensor produced by
            preprocessing.py.

    Returns:
        tuple:
            boxes        (list[list[int]]): Detected bounding boxes, each
                         as [x1, y1, x2, y2] in 640×640 pixel space.
            confidences  (list[float])    : Confidence score for every
                         surviving box (same order as boxes).
            inference_time_ms (float)     : Raw model execution time in ms.
            fps               (float)     : Frames per second (1000 /
                                           inference_time_ms).
    """


    # 1. Run the ONNX session and measure wall-clock inference time
    input_name = session.get_inputs()[0].name

    t_start = time.perf_counter()
    raw_outputs = session.run(None, {input_name: preprocessed_frame})
    t_end = time.perf_counter()

    inference_time_ms = (t_end - t_start) * 1000.0          # ms
    fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0


    # 2. Parse the YOLOv8n output tensor [1, 5, 8400]
    #    Transpose to [8400, 5] so each row = one anchor proposal.
    output_tensor = raw_outputs[0]            # shape: (1, 5, 8400)
    output_tensor = output_tensor[0]          # shape: (5, 8400)
    output_tensor = output_tensor.T           # shape: (8400, 5)
    #   columns: [cx, cy, w, h, conf]


    # 3. Confidence filtering
    confidences_raw = output_tensor[:, 4]
    mask = confidences_raw >= CONFIDENCE_THRESHOLD
    filtered = output_tensor[mask]            # shape: (N, 5)

    if filtered.shape[0] == 0:
        return [], [], inference_time_ms, fps


    # 4. Convert [cx, cy, w, h] → [x1, y1, w, h] for NMSBoxes
    #    All values are already in 640×640 pixel space (the model was exported with pixel-unit outputs, not 0-1 normalised).
    cx  = filtered[:, 0]
    cy  = filtered[:, 1]
    w   = filtered[:, 2]
    h   = filtered[:, 3]
    confs = filtered[:, 4].tolist()

    x1 = (cx - w / 2.0)
    y1 = (cy - h / 2.0)

    # cv2.dnn.NMSBoxes expects a list of [x, y, w, h] (top-left + size)
    nms_boxes = [
        [int(x1[i]), int(y1[i]), int(w[i]), int(h[i])]
        for i in range(len(confs))
    ]


    # 5. Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=confs,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )


    # 6. Collect surviving detections as [x1, y1, x2, y2]
    final_boxes: list[list[int]] = []
    final_confidences: list[float] = []

    if len(indices) > 0:
        # OpenCV 4.x returns shape (N, 1); flatten to 1-D
        indices = indices.flatten()
        for idx in indices:
            bx, by, bw, bh = nms_boxes[idx]
            x2 = bx + bw
            y2 = by + bh
            # Clamp to valid frame dimensions
            x1c = max(0, bx)
            y1c = max(0, by)
            x2c = min(INPUT_SIZE[0] - 1, x2)
            y2c = min(INPUT_SIZE[1] - 1, y2)
            final_boxes.append([x1c, y1c, x2c, y2c])
            final_confidences.append(round(confs[idx], 4))

    return final_boxes, final_confidences, inference_time_ms, fps