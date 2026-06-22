import cv2
import numpy as np

def draw_results(frame: np.ndarray, detections: list[dict], roi_manager=None, color_map: dict = None, mask_alpha: float = 0.4) -> np.ndarray:
    """Draws detections and ROI boundaries onto the frame.
    
    Args:
        frame: Original input frame
        detections: List of detection dictionaries
        roi_manager: Optional ROIManager instance to draw ROI boundaries
        color_map: Map of category name to BGR color lists
        mask_alpha: Opacity of the segmentation mask overlay
    """
    out = frame.copy()

    # 1. Draw ROI boundaries if provided
    if roi_manager:
        out = roi_manager.draw_rois(out)

    # 2. Draw detections
    for det in detections:
        name = det["class_name"]
        conf = det["confidence"]
        bbox = [int(v) for v in det["bbox"]]
        mask = det.get("mask")
        area = det.get("area_px", 0)
        roi_name = det.get("roi_name", "")

        # Default to gray if class not in color map
        color = (128, 128, 128)
        if color_map and name in color_map:
            color = tuple(color_map[name])

        # Overlay segmentation mask
        if mask is not None:
            overlay = out.copy()
            overlay[mask > 0] = color
            cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0, out)

        # Draw bounding box
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Build text label (name, conf, area, and roi_name)
        label = f"{name} {conf:.2f} ({area}px)"
        if roi_name:
            label += f" inside {roi_name}"

        # Draw text label above the bounding box
        cv2.putText(
            out,
            label,
            (bbox[0], max(bbox[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    return out
