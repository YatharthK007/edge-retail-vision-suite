import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless 
import matplotlib.pyplot as plt

import config
import preprocessing
import inference
import utils
import logger



# Helpers

def _save_final_heatmap(heatmap_array: np.ndarray, out_path: str) -> None:
    """
    Normalise the accumulated heatmap and save it as a colourised PNG using
    Matplotlib so the saved image matches what was shown on-screen.

    Args:
        heatmap_array (np.ndarray): Float32 accumulation array (H × W).
        out_path      (str)       : Destination file path (created if absent).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title("ERVS — Final Session Heatmap", fontsize=13, fontweight="bold")
    ax.axis("off")

    max_val = heatmap_array.max()
    display = heatmap_array / max_val if max_val > 0 else heatmap_array

    im = ax.imshow(display, cmap="jet", interpolation="bilinear", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Relative occupancy")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[main] Heatmap saved → {out_path}")



# Entry point

def main() -> None:
    """
    Run the ERVS real-time analytics pipeline.

    Pipeline summary per frame:
        1.  Read raw frame from VideoCapture.
        2.  Honour FRAME_SKIP to reduce load on the Jetson Nano.
        3.  Preprocess frame → (1, 3, 640, 640) float32 tensor.
        4.  Run ONNX inference → boxes in 640×640 space + timing.
        5.  Restore boxes to original frame coordinates.
        6.  Draw detections, update + draw tripwire, update heatmap
            (optionally overlay), update + draw dwell zones, draw FPS.
        7.  Log events (crossings, FPS, dwell) via logger.py.
        8.  Display annotated frame; handle keyboard shortcuts.
        9.  On exit: save final heatmap PNG, log session summary.

    Keyboard controls while the window is open:
        q — quit
        h — toggle heatmap overlay on/off
        s — save a screenshot to runs/screenshot.jpg
    """

    #  0. Initialise logger 
    logger.setup_logger()

    #  1. Load ONNX model 
    session = inference.load_model(config.MODEL_PATH)

    #  2. Open video source
    print(f"[main] Opening source: {config.INPUT_SOURCE}")
    cap = cv2.VideoCapture(config.INPUT_SOURCE)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"[main] Cannot open input source: {config.INPUT_SOURCE}\n"
            "       Check INPUT_SOURCE in config.py."
        )

    # Read one frame to learn the native resolution
    ret, probe = cap.read()
    if not ret:
        raise RuntimeError("[main] Video source produced no frames.")

    frame_h, frame_w = probe.shape[:2]
    print(f"[main] Video resolution: {frame_w}×{frame_h}")

    # Rewind so the probe frame is processed normally in the loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #  3. Initialise state variables
    heatmap_array: np.ndarray = np.zeros((frame_h, frame_w), dtype=np.float32)

    entry_count:  int  = 0
    exit_count:   int  = 0
    prev_entry:   int  = 0
    prev_exit:    int  = 0

    tracked_centroids: dict = {}
    next_id:           int  = 1
    dwell_tracker:     dict = {}

    frame_count:  int  = 0
    show_heatmap: bool = False

    # Last known FPS/ms — reused on skipped frames so the overlay stays live
    last_fps:        float = 0.0
    last_inf_ms:     float = 0.0
    last_orig_boxes: list  = []
    last_confs:      list  = []

    # Ensure output directory exists
    os.makedirs(os.path.join(config.PROJECT_ROOT, "runs"), exist_ok=True)

    print("[main] Starting main loop.  Press  q=quit  h=heatmap  s=screenshot\n")

    cv2.namedWindow("ERVS - Edge Retail Vision Suite", cv2.WINDOW_NORMAL)

    #  4. Main loop
    while True:

        #  4-1. Read frame
        ret, frame = cap.read()
        if not ret:
            print("[main] End of video stream.")
            break

        frame_count += 1

        #  4-2. Frame-skip logic
        # On skipped frames we still annotate and display using the most recent inference results — the video never freezes.
        run_inference_this_frame = (
            config.FRAME_SKIP <= 1
            or frame_count % config.FRAME_SKIP == 0
        )

        if run_inference_this_frame:

            #  4-3. Preprocess 
            # Returns: padded tensor (1,3,640,640), x-scale, y-scale,
            #          x-padding, y-padding
            pframe, sx, sy, px, py = preprocessing.preprocess_frame(frame)

            #  4-4. Inference 
            boxes_640, confs, inference_ms, fps = inference.run_inference(
                session, pframe
            )

            # Cache for skipped frames
            last_fps    = fps
            last_inf_ms = inference_ms
            last_confs  = confs

            #  4-5. Coordinate restoration
            # From 640×640 letterbox space → original frame pixel space.
            # Formula per component: orig = (letterbox_coord - padding) / scale
            orig_boxes: list = []
            for x1, y1, x2, y2 in boxes_640:
                ox1 = int((x1 - px) / sx)
                oy1 = int((y1 - py) / sy)
                ox2 = int((x2 - px) / sx)
                oy2 = int((y2 - py) / sy)

                # Clamp to frame boundaries (avoids out-of-bounds draw calls)
                ox1 = max(0, min(ox1, frame_w - 1))
                oy1 = max(0, min(oy1, frame_h - 1))
                ox2 = max(0, min(ox2, frame_w - 1))
                oy2 = max(0, min(oy2, frame_h - 1))

                orig_boxes.append([ox1, oy1, ox2, oy2])

            last_orig_boxes = orig_boxes

            #  4-7. Tripwire tracker 
            entry_count, exit_count, tracked_centroids, next_id = (
                utils.check_tripwire(
                    orig_boxes,
                    tracked_centroids,
                    next_id,
                    entry_count,
                    exit_count,
                    config.TRIPWIRE_Y,
                )
            )

            #  4-8. Tripwire logging 
            if entry_count > prev_entry:
                logger.log_entry(entry_count)
                prev_entry = entry_count

            if exit_count > prev_exit:
                logger.log_exit(exit_count)
                prev_exit = exit_count

            #  4-10. Heatmap accumulation
            utils.update_heatmap(heatmap_array, orig_boxes)

            #  4-12. Dwell zones 
            dwell_tracker = utils.check_dwell_zones(
                orig_boxes,
                config.DWELL_ZONES,
                dwell_tracker,
                frame_count,
                fps,
            )

            #  4-15. Periodic FPS logging
            if frame_count % 30 == 0:
                logger.log_fps(fps, inference_ms)

        else:
            # Skipped frame — reuse last inference results
            orig_boxes = last_orig_boxes
            fps        = last_fps

        #  4-6. Draw detections 
        utils.draw_detections(frame, orig_boxes, last_confs)

        #  4-9. Draw tripwire 
        utils.draw_tripwire(frame, config.TRIPWIRE_Y, entry_count, exit_count)

        #  4-11. Heatmap overlay (toggleable)
        if show_heatmap:
            frame = utils.render_heatmap_overlay(frame, heatmap_array)

        #  4-13. Draw dwell zones 
        utils.draw_dwell_zones(frame, config.DWELL_ZONES, dwell_tracker)

        #  4-14. Draw FPS 
        utils.draw_fps(frame, last_fps)

        #  4-16. Display 
        cv2.imshow("ERVS - Edge Retail Vision Suite", frame)

        #  4-17. Keyboard controls 
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[main] 'q' pressed — exiting.")
            break

        elif key == ord("h"):
            show_heatmap = not show_heatmap
            state = "ON" if show_heatmap else "OFF"
            print(f"[main] Heatmap overlay toggled {state}.")

        elif key == ord("s"):
            screenshot_path = os.path.join(
                config.PROJECT_ROOT, "runs", "screenshot.jpg"
            )
            cv2.imwrite(screenshot_path, frame)
            print(f"[main] Screenshot saved → {screenshot_path}")

    #  5. Teardown
    cap.release()
    cv2.destroyAllWindows()
    print("[main] VideoCapture released, windows closed.")

    # 5a. Save final heatmap PNG 
    heatmap_out = os.path.join(config.PROJECT_ROOT, "runs", "heatmap_final.png")
    _save_final_heatmap(heatmap_array, heatmap_out)

    # 5b. Dwell logging
    for zone_name, data in dwell_tracker.items():
        seconds = data["seconds"] if isinstance(data, dict) else float(data)
        logger.log_dwell(zone_name, seconds)

    # 5c. Session summary 
    logger.log_session_end(entry_count, exit_count)

    print(
        f"\n[main] Session complete.\n"
        f"       Frames processed : {frame_count}\n"
        f"       Total entries    : {entry_count}\n"
        f"       Total exits      : {exit_count}\n"
        f"       Log              : {config.LOG_FILE_PATH}\n"
    )



if __name__ == "__main__":
    main()