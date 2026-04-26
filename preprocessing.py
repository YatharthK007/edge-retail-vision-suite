import cv2
import numpy as np

import config



# Public API

def preprocess_frame(
    frame: np.ndarray,
) -> tuple[np.ndarray, float, float, int, int]:
    """Letterbox-resize a raw BGR frame and normalise it for model inference.

    Letterboxing preserves the original aspect ratio by scaling the frame so
    its longest side equals the model's expected input dimension, then padding
    the shorter axis symmetrically with mid-grey (128).  This avoids the
    distortion that plain stretching would introduce and matches the
    preprocessing used during YOLO training.

    Pipeline
    --------
    1. Compute a single uniform scale factor (no distortion).
    2. Resize the frame with that factor.
    3. Pad width or height to reach exactly ``config.INPUT_SIZE``.
    4. Convert BGR → RGB (ONNX models expect RGB).
    5. Normalise uint8 [0, 255] → float32 [0.0, 1.0].
    6. Transpose HWC → CHW and add a batch dimension → (1, 3, H, W).

    Parameters
    ----------
    frame : np.ndarray
        Raw BGR frame from ``cv2.VideoCapture.read()``, shape (H, W, 3),
        dtype uint8.

    Returns
    -------
    blob : np.ndarray
        Ready-to-infer array, shape (1, 3, INPUT_H, INPUT_W), dtype float32.
    scale_x : float
        Ratio of model-input width to original frame width *after* accounting
        for padding.  Used by ``restore_coordinates`` to invert the transform.
    scale_y : float
        Same as ``scale_x`` but for the vertical axis.
    pad_x : int
        Number of pixels added to the LEFT side of the letterboxed image.
        (Right padding = total pad - pad_x, but symmetric so both are equal.)
    pad_y : int
        Number of pixels added to the TOP of the letterboxed image.
    """
    target_w, target_h = config.INPUT_SIZE          # e.g. 640, 640
    orig_h, orig_w = frame.shape[:2]


    # 1. Uniform scale: fit the frame inside target_w × target_h

    scale = min(target_w / orig_w, target_h / orig_h)

    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))

    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)


    # 2. Pad to exact target dimensions (symmetric, mid-grey = 128)

    pad_total_x = target_w - scaled_w       # total horizontal padding needed
    pad_total_y = target_h - scaled_h       # total vertical   padding needed

    pad_x = pad_total_x // 2               # left  pad  (right  gets the remainder)
    pad_y = pad_total_y // 2               # top   pad  (bottom gets the remainder)

    letterboxed = cv2.copyMakeBorder(
        resized,
        top=pad_y,
        bottom=pad_total_y - pad_y,
        left=pad_x,
        right=pad_total_x - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(128, 128, 128),             # neutral grey avoids bias
    )


    # 3. Colour-space, normalisation, layout conversion

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

    # Normalise to [0, 1] — keeps values in a single float32 op.
    normalised = rgb.astype(np.float32) / 255.0

    # HWC → CHW, then add batch dimension: (H,W,3) → (1,3,H,W)
    blob = np.transpose(normalised, (2, 0, 1))[np.newaxis, ...]


    # 4. Compute scale factors for the inverse transform
    # These describe how model-space coords map back to original-frame coords BEFORE accounting for padding (padding is removed first).
    # After removing padding the active image is (scaled_w × scaled_h), which was produced by uniformly scaling the original by `scale`.
    # So: orig_coord = (model_coord - pad) / scale
    # Expressed as separate x/y factors for restore_coordinates:
    scale_x = scale     # same in both axes (uniform scaling)
    scale_y = scale

    return blob, scale_x, scale_y, pad_x, pad_y


def restore_coordinates(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Map bounding boxes from letterboxed model-input space to original frame space.

    Applies the exact inverse of the letterbox transform performed by
    ``preprocess_frame``:

        original_x = (model_x - pad_x) / scale_x
        original_y = (model_y - pad_y) / scale_y

    Coordinates are clipped to the valid pixel range so that partially
    out-of-frame boxes do not cause downstream index errors.

    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes in letterboxed (640×640) space.
        Shape: (N, 4) with columns [x1, y1, x2, y2] in pixel coordinates.
        Pass an empty array (shape (0, 4)) if the model returned no detections.
    scale_x : float
        ``scale_x`` returned by ``preprocess_frame`` for this frame.
    scale_y : float
        ``scale_y`` returned by ``preprocess_frame`` for this frame.
    pad_x : int
        ``pad_x`` returned by ``preprocess_frame`` for this frame.
    pad_y : int
        ``pad_y`` returned by ``preprocess_frame`` for this frame.

    Returns
    -------
    np.ndarray
        Bounding boxes in original frame pixel coordinates.
        Shape: (N, 4), dtype float32, columns [x1, y1, x2, y2].
        Returns an empty (0, 4) array if ``boxes`` was empty.
    """
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32).copy()

    # Remove letterbox padding then invert the uniform scale.
    boxes[:, 0] = (boxes[:, 0] - pad_x) / scale_x   # x1
    boxes[:, 1] = (boxes[:, 1] - pad_y) / scale_y   # y1
    boxes[:, 2] = (boxes[:, 2] - pad_x) / scale_x   # x2
    boxes[:, 3] = (boxes[:, 3] - pad_y) / scale_y   # y2

    # Clip to [0, large number]; inference.py clips to frame dims once it has the actual frame shape, but a coarse non-negative clip is useful here.
    boxes = np.clip(boxes, 0, None)

    return boxes