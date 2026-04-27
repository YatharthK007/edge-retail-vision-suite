import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import (
    BBOX_COLOR,
    BBOX_THICKNESS,
    TRIPWIRE_COLOR,
    TRIPWIRE_THICKNESS,
    DWELL_ZONE_COLOR,
    DWELL_ZONE_THICKNESS,
    HEATMAP_ALPHA,
)

# Type aliases for readability
Box           = List[int]                          # [x1, y1, x2, y2]
Centroid      = Tuple[int, int]                    # (cx, cy)
TrackedDict   = Dict[int, Centroid]                # {track_id: (cx, cy)}
DwellTracker  = Dict[str, float]                   # {zone_name: frame_count}
ZoneDict      = Dict[str, List[int]]               # {zone_name: [x1,y1,x2,y2]}

# Maximum pixel distance below which two centroids are considered the same
# person across consecutive frames.
_MAX_MATCH_DIST: float = 50.0



# 1. draw_detections

def draw_detections(
    frame: np.ndarray,
    boxes: List[Box],
    confidences: List[float],
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels for every detected person.

    Each box is drawn as a green rectangle (colour from config.BBOX_COLOR).
    A filled label strip above the box shows "Person: XX%" so the text is
    always readable regardless of background.

    Args:
        frame       (np.ndarray)   : BGR frame to annotate (modified in place
                                     and also returned for chaining).
        boxes       (List[Box])    : Detected bounding boxes, each as
                                     [x1, y1, x2, y2] in frame pixel space.
        confidences (List[float])  : Confidence score per box (0.0 – 1.0).

    Returns:
        np.ndarray: The annotated frame (same object as `frame`).
    """
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box

        # Bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

        # Confidence label
        label      = f"Person: {conf * 100:.1f}%"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Filled background strip so text is legible over any colour
        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = max(y1, th + baseline + 4)
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + tw + 4, label_y2),
            BBOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, label_y2 - baseline - 2),
            font, font_scale,
            (0, 0, 0),          # black text on green background
            thickness,
            cv2.LINE_AA,
        )

    return frame



# 2. draw_tripwire

def draw_tripwire(
    frame: np.ndarray,
    y_coord: int,
    entry_count: int,
    exit_count: int,
) -> np.ndarray:
    """
    Draw the virtual tripwire line and the live entry/exit counter.

    A horizontal red line (colour from config.TRIPWIRE_COLOR) is drawn at
    `y_coord` spanning the full frame width.  An "IN: X | OUT: Y" counter
    is rendered at the top-left with a semi-transparent dark background so
    it remains readable over any scene.

    Args:
        frame       (np.ndarray) : BGR frame to annotate.
        y_coord     (int)        : Y-pixel position of the tripwire.
        entry_count (int)        : Cumulative persons counted entering.
        exit_count  (int)        : Cumulative persons counted exiting.

    Returns:
        np.ndarray: The annotated frame.
    """
    frame_w = frame.shape[1]

    # Tripwire line
    cv2.line(
        frame,
        (0, y_coord), (frame_w, y_coord),
        TRIPWIRE_COLOR,
        TRIPWIRE_THICKNESS,
    )

    # Counter text with dark backing rectangle
    counter_text = f"IN: {entry_count}  |  OUT: {exit_count}"
    font         = cv2.FONT_HERSHEY_SIMPLEX
    font_scale   = 0.7
    thickness    = 2
    (tw, th), baseline = cv2.getTextSize(counter_text, font, font_scale, thickness)

    pad = 6
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (tw + pad * 2, th + baseline + pad * 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, counter_text,
        (pad, th + pad),
        font, font_scale,
        (255, 255, 255),        # white text
        thickness,
        cv2.LINE_AA,
    )

    return frame



# 3. check_tripwire

def check_tripwire(
    boxes: List[Box],
    tracked_centroids: TrackedDict,
    next_id: int,
    entry_count: int,
    exit_count: int,
    y_coord: int,
) -> Tuple[int, int, TrackedDict, int]:
    """
    Simple Euclidean-distance centroid tracker with tripwire crossing logic.

    Algorithm (per frame):
        1. Compute the centre (cx, cy) for every detected box.
        2. For each new centroid, find the closest existing track in
           `tracked_centroids` (by Euclidean distance).
        3. If the closest match is within _MAX_MATCH_DIST pixels, the
           detection is assigned that track ID and the stored centroid is
           updated to the new position.
        4. If no match is close enough, a brand-new track ID is minted and
           `next_id` is incremented.
        5. Crossing detection compares the previous cy (before update) to
           the new cy:
               • prev_cy <  y_coord AND new_cy >= y_coord  →  ENTRY (+1)
               • prev_cy >  y_coord AND new_cy <= y_coord  →  EXIT  (+1)
        6. Tracks with no matching detection this frame are dropped to
           prevent ghost tracks from accumulating over time.

    Args:
        boxes             (List[Box])   : Current-frame boxes [x1,y1,x2,y2].
        tracked_centroids (TrackedDict) : {track_id: (cx, cy)} from last frame.
        next_id           (int)         : Next available track ID.
        entry_count       (int)         : Running entry total.
        exit_count        (int)         : Running exit total.
        y_coord           (int)         : Y-coordinate of the tripwire.

    Returns:
        Tuple:
            updated_entry_count   (int)
            updated_exit_count    (int)
            new_tracked_centroids (TrackedDict)  – only live tracks retained.
            updated_next_id       (int)
    """
    new_tracked: TrackedDict = {}

    # Compute current-frame centroids from bounding boxes
    current_centroids: List[Centroid] = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        current_centroids.append((cx, cy))

    # Convert existing tracks to a list for sequential matching
    existing_ids       = list(tracked_centroids.keys())
    existing_centroids = [tracked_centroids[tid] for tid in existing_ids]
    matched_existing   = set()      # indices of already-claimed tracks

    for new_cx, new_cy in current_centroids:
        best_id   : int   = -1
        best_dist : float = float("inf")

        for i, (ex_cx, ex_cy) in enumerate(existing_centroids):
            if i in matched_existing:
                continue            # already taken by another detection
            dist = math.hypot(new_cx - ex_cx, new_cy - ex_cy)
            if dist < best_dist:
                best_dist = dist
                best_id   = i

        if best_id != -1 and best_dist <= _MAX_MATCH_DIST:
            #  Matched to an existing track 
            track_id         = existing_ids[best_id]
            _, prev_cy       = existing_centroids[best_id]
            matched_existing.add(best_id)

            # Crossing detection
            if prev_cy < y_coord <= new_cy:     # moved downward across line
                entry_count += 1
            elif prev_cy > y_coord >= new_cy:   # moved upward across line
                exit_count += 1

            new_tracked[track_id] = (new_cx, new_cy)

        else:
            #  New track 
            new_tracked[next_id] = (new_cx, new_cy)
            next_id += 1

    return entry_count, exit_count, new_tracked, next_id



# 4. update_heatmap

def update_heatmap(
    heatmap_array: np.ndarray,
    boxes: List[Box],
) -> np.ndarray:
    """
    Accumulate detection presence into the heatmap array.

    For every detected bounding box, the region of `heatmap_array` that
    the box occupies is incremented by 1.0 using NumPy slice assignment —
    a single vectorized operation per box, with no nested Python loops.

    Args:
        heatmap_array (np.ndarray): Float32 accumulation array shaped
                                    (frame_height, frame_width).  Modified
                                    in place and also returned for chaining.
        boxes         (List[Box]) : Detected boxes as [x1, y1, x2, y2].

    Returns:
        np.ndarray: Updated heatmap_array (same object).
    """
    h, w = heatmap_array.shape[:2]

    for x1, y1, x2, y2 in boxes:
        # Clamp coordinates so slices never exceed array bounds
        rx1 = max(0, min(x1, w - 1))
        ry1 = max(0, min(y1, h - 1))
        rx2 = max(0, min(x2, w))
        ry2 = max(0, min(y2, h))

        if rx2 > rx1 and ry2 > ry1:                    # guard empty slice
            heatmap_array[ry1:ry2, rx1:rx2] += 1.0     # vectorized NumPy slice

    return heatmap_array



# 5. render_heatmap_overlay

def render_heatmap_overlay(
    frame: np.ndarray,
    heatmap_array: np.ndarray,
) -> np.ndarray:
    """
    Normalise the heatmap accumulation array and blend it over the frame.

    Pipeline:
        1. Normalise heatmap_array to [0, 255] uint8.
        2. Apply cv2.COLORMAP_JET to produce a false-colour heat image.
        3. Blend with the original frame using cv2.addWeighted at the alpha
           value defined in config.HEATMAP_ALPHA.

    If `heatmap_array` is all zeros (no detections yet), the original frame
    is returned unchanged to avoid a divide-by-zero artefact.

    Args:
        frame         (np.ndarray): BGR frame to blend onto.
        heatmap_array (np.ndarray): Float32 accumulation array, same H×W as
                                    frame.

    Returns:
        np.ndarray: Blended BGR frame.
    """
    max_val = heatmap_array.max()
    if max_val == 0.0:
        return frame                            # nothing accumulated yet

    # Normalise → uint8
    normalised = (heatmap_array / max_val * 255.0).astype(np.uint8)

    # False-colour map
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

    # Resize coloured map to match frame if dimensions differ
    if coloured.shape[:2] != frame.shape[:2]:
        coloured = cv2.resize(
            coloured,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Alpha blend: output = alpha * heatmap + (1 - alpha) * frame
    blended = cv2.addWeighted(coloured, HEATMAP_ALPHA, frame, 1.0 - HEATMAP_ALPHA, 0)
    return blended



# 6. check_dwell_zones

def check_dwell_zones(
    boxes: List[Box],
    zone_dict: ZoneDict,
    dwell_tracker: DwellTracker,
    frame_count: int,
    fps: float,
) -> DwellTracker:
    """
    Accumulate the number of frames each zone has been occupied and derive
    dwell time in seconds.

    For every detected box, its centre point is tested against every zone
    rectangle.  Whenever a centre falls inside a zone, that zone's frame
    counter is incremented by 1.  Dividing accumulated frames by `fps`
    gives total dwell time in seconds, stored under the "seconds" key.

    The `dwell_tracker` dict structure:
        {
            "zone_name": {
                "frames":  int,    # raw accumulated frame count
                "seconds": float,  # frames / fps
            },
            …
        }

    Zone keys absent on first call are initialised automatically.

    Args:
        boxes         (List[Box])    : Current-frame boxes [x1,y1,x2,y2].
        zone_dict     (ZoneDict)     : Named zone rectangles from config.py.
        dwell_tracker (DwellTracker) : Persistent accumulator dict (mutated
                                       and returned).
        frame_count   (int)          : Current video frame index (available
                                       to callers for logging).
        fps           (float)        : Video / inference FPS used to convert
                                       frame counts to seconds.

    Returns:
        DwellTracker: The updated accumulator dict.
    """
    safe_fps = fps if fps > 0.0 else 1.0    # guard against zero-division

    for zone_name, (zx1, zy1, zx2, zy2) in zone_dict.items():
        # Initialise zone entry on first encounter
        if zone_name not in dwell_tracker:
            dwell_tracker[zone_name] = {"frames": 0, "seconds": 0.0}

        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Point-in-rectangle test (inclusive boundaries)
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                dwell_tracker[zone_name]["frames"]  += 1
                dwell_tracker[zone_name]["seconds"]  = (
                    dwell_tracker[zone_name]["frames"] / 25.0
                )

    return dwell_tracker



# 7. draw_fps

def draw_fps(
    frame: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Render the current FPS reading at the top-right corner of the frame.

    A semi-transparent dark backing rectangle is drawn behind the text so
    it remains legible regardless of the video content underneath.

    Args:
        frame (np.ndarray): BGR frame to annotate.
        fps   (float)     : Current frames-per-second value.

    Returns:
        np.ndarray: The annotated frame.
    """
    label      = f"FPS: {fps:.1f}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    frame_w = frame.shape[1]
    pad     = 6
    x_start = frame_w - tw - pad * 2
    y_end   = th + baseline + pad * 2

    # Semi-transparent backing rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, 0), (frame_w, y_end), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, label,
        (x_start + pad, th + pad),
        font, font_scale,
        (0, 255, 255),          # yellow text
        thickness,
        cv2.LINE_AA,
    )

    return frame



# Bonus: draw_dwell_zones  (called by main.py to annotate zone rectangles)

def draw_dwell_zones(
    frame: np.ndarray,
    zone_dict: ZoneDict,
    dwell_tracker: DwellTracker,
) -> np.ndarray:
    """
    Draw each dwell zone as a labelled rectangle on the frame.

    The label shows the zone name and accumulated dwell time in seconds so
    the operator can see live occupancy at a glance.

    Args:
        frame         (np.ndarray)   : BGR frame to annotate.
        zone_dict     (ZoneDict)     : Named zone rectangles from config.py.
        dwell_tracker (DwellTracker) : Current accumulator from
                                       check_dwell_zones().

    Returns:
        np.ndarray: The annotated frame.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness  = 1

    for zone_name, (zx1, zy1, zx2, zy2) in zone_dict.items():
        # Zone bounding rectangle
        cv2.rectangle(
            frame,
            (zx1, zy1), (zx2, zy2),
            DWELL_ZONE_COLOR,
            DWELL_ZONE_THICKNESS,
        )

        # Label: "zone_name  X.Xs"
        seconds = dwell_tracker.get(zone_name, {}).get("seconds", 0.0)
        label   = f"{zone_name}: {seconds:.1f}s"

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        lx1 = zx1
        ly1 = max(zy1 - th - baseline - 4, 0)
        ly2 = max(zy1, th + baseline + 4)

        # Filled label background
        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx1 + tw + 4, ly2),
            DWELL_ZONE_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (lx1 + 2, ly2 - baseline - 2),
            font, font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return frame