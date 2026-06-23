import cv2
import numpy as np
import yaml
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ROIManager:
    """Manages Regions of Interest (ROIs), handles interactive manual selection, and filters detections."""
    def __init__(self, roi_yaml_path: str = "config/roi.yaml"):
        self.roi_path = Path(roi_yaml_path)
        self.rois = []
        self.load_rois()

    def load_rois(self):
        """Loads previous ROI definitions from YAML configuration."""
        if not self.roi_path or not self.roi_path.exists():
            return

        try:
            with open(self.roi_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if data and "rois" in data:
                self.rois = []
                for r in data["rois"]:
                    name = r.get("name", "unnamed")
                    pts = r.get("polygon", [])
                    if len(pts) >= 3:
                        poly = np.array(pts, dtype=np.int32)
                        self.rois.append({
                            "name": name,
                            "polygon": poly
                        })
                logger.info(f"Loaded previous ROI from {self.roi_path.absolute()}")
        except Exception as e:
            logger.error(f"Error loading ROI configuration: {e}")

    def save_rois(self):
        """Saves current ROI definitions to YAML configuration."""
        if not self.roi_path or not self.rois:
            return

        try:
            self.roi_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "rois": [
                    {
                        "name": r["name"],
                        "polygon": r["polygon"].tolist()
                    }
                    for r in self.rois
                ]
            }
            with open(self.roi_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            logger.info(f"Saved manual ROI to {self.roi_path.absolute()}")
        except Exception as e:
            logger.error(f"Error saving ROI: {e}")

    def select_roi_interactively(self, frame: np.ndarray):
        """Opens a window allowing the user to manually click points to define a polygon ROI."""
        logger.info("==========================================================")
        logger.info("ROI SELECTION WINDOW OPENED.")
        logger.info("Instructions:")
        logger.info(" - Left Click: Add a polygon vertex point")
        logger.info(" - ENTER or SPACE: Finish drawing and proceed")
        logger.info(" - 'C' key: Clear current drawing to restart")
        logger.info(" - ESC or 'Q' key: Skip ROI selection (process entire frame)")
        logger.info("==========================================================")

        points = []
        window_name = "Draw ROI (Click points, ENTER=Save, C=Clear, ESC=Skip)"
        display_frame = frame.copy()
        
        # If we have a previously loaded ROI, overlay it as a preview
        has_previous = len(self.rois) > 0
        if has_previous:
            # Draw previous ROI in a dashed/dotted or thin blue outline
            cv2.polylines(display_frame, [self.rois[0]["polygon"]], isClosed=True, color=(255, 128, 0), thickness=2)
            cv2.putText(
                display_frame,
                "Previous ROI (Press ENTER to reuse, or click to draw new)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 128, 0),
                2,
                cv2.LINE_AA
            )

        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                # Render the current drawing on a clean frame
                self._draw_temp_polygon(display_frame, frame, points)
                cv2.imshow(window_name, display_frame)

        try:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, display_frame)

            while True:
                key = cv2.waitKey(20) & 0xFF
                
                # Enter or Space key to finish
                if key == 13 or key == 32:
                    if len(points) >= 3:
                        self.rois = [{
                            "name": "",
                            "polygon": np.array(points, dtype=np.int32)
                        }]
                        self.save_rois()
                        break
                    elif has_previous and not points:
                        # User didn't click anything but pressed Enter, reuse the cached ROI
                        logger.info("Reusing previous manual ROI configuration.")
                        break
                    else:
                        logger.warning("Please click at least 3 points before pressing Enter.")

                # Clear points
                elif key == ord('c') or key == ord('C'):
                    points.clear()
                    display_frame = frame.copy()
                    cv2.imshow(window_name, display_frame)
                    logger.info("Cleared drawn ROI points.")

                # Esc or Q to skip
                elif key == 27 or key == ord('q') or key == ord('Q'):
                    logger.info("ROI selection skipped. System will run on full frame.")
                    self.rois = []
                    break

            cv2.destroyWindow(window_name)
        except cv2.error as e:
            logger.warning(f"Interactive ROI selection window could not be opened: {e}. Reusing loaded ROI or proceeding.")

    def _draw_temp_polygon(self, draw_img: np.ndarray, orig_img: np.ndarray, pts: list[tuple[int, int]]):
        """Draws temporary points and lines for the user during interactive selection."""
        # Reset image to original frame state
        draw_img[:] = orig_img[:]
        
        # Draw all clicked vertex points
        for pt in pts:
            cv2.circle(draw_img, pt, 5, (0, 0, 255), -1)
            
        # Draw lines connecting the vertices
        if len(pts) > 1:
            poly_pts = np.array(pts, dtype=np.int32)
            cv2.polylines(draw_img, [poly_pts], isClosed=False, color=(0, 255, 0), thickness=2)

    def filter_detections(self, detections: list[dict]) -> list[dict]:
        """Filters detections to keep only those whose centroid is inside the configured ROI.
        
        If no ROI is defined, returns all detections.
        """
        if not self.rois:
            return detections

        filtered = []
        for det in detections:
            polygon = det.get("polygon")
            bbox = det.get("bbox")

            # Determine centroid
            cX, cY = None, None
            if polygon is not None and len(polygon) > 0:
                M = cv2.moments(polygon)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
            
            if cX is None or cY is None:
                cX = int((bbox[0] + bbox[2]) / 2)
                cY = int((bbox[1] + bbox[3]) / 2)

            # Check point polygon overlap
            in_any_roi = False
            for roi in self.rois:
                dist = cv2.pointPolygonTest(roi["polygon"].astype(np.float32), (float(cX), float(cY)), False)
                if dist >= 0:
                    in_any_roi = True
                    det["roi_name"] = roi["name"]
                    break

            if in_any_roi:
                filtered.append(det)

        return filtered

    def draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Draws current ROI outline on output frames."""
        out = frame.copy()
        for roi in self.rois:
            cv2.polylines(out, [roi["polygon"]], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(
                out,
                f"ROI: {roi['name']}",
                (roi["polygon"][0][0], max(roi["polygon"][0][1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
        return out
