import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SpillDetector:
    """Wrapper for YOLO segmentation and tracking model to detect spills and track instances."""
    def __init__(self, weights_path: str, device: str = "cuda", imgsz: int = 640, conf: float = 0.45, iou: float = 0.45):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.weights_path.absolute()}")

        logger.info(f"Loading YOLO model from {self.weights_path} on device {device}")
        
        # Load YOLO model
        self.model = YOLO(str(self.weights_path))
        
        # Device transfer is only necessary for PyTorch (.pt) weights.
        # TensorRT engines (.engine) are compiled directly for GPU execution.
        if self.weights_path.suffix == ".pt":
            self.model.to(device)
            
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        # Run model warmup to ensure first frame inference is fast
        logger.info("Warming up model...")
        dummy = np.zeros((imgsz, imgsz, 3), dtype="uint8")
        self.predict(dummy)
        logger.info("Warmup complete.")

    def predict(self, frame: np.ndarray):
        """Perform raw YOLO segmentation inference (used for warmup)."""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        return results[0]

    def track(self, frame: np.ndarray, persist: bool = True):
        """Perform YOLO tracking and segmentation inference on the frame sequence."""
        results = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            persist=persist,
            verbose=False
        )
        return results[0]

    def detect_spills(self, frame: np.ndarray, min_area_px: int = 500, class_names: dict = None) -> list[dict]:
        """Runs tracking and postprocesses outputs into structured dictionaries.
        
        Each detection contains:
            class_id (int)
            class_name (str)
            confidence (float)
            bbox (list[float]): [x1, y1, x2, y2]
            mask (np.ndarray): Binary segmentation mask matching frame dimensions
            area_px (int): Pixel area of the mask
            polygon (np.ndarray): Simplified polygon vertices outlining the contour
            track_id (int | None): Unique object track ID if tracking is active
        """
        result = self.track(frame, persist=True)
        detections = []

        if result.masks is None or result.boxes is None:
            return detections

        h, w = frame.shape[:2]
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            # Extract track ID if assigned by tracker
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            # Extract raw mask and resize to original frame dimensions
            mask_data = result.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype("uint8") * 255

            # Filter out detections below the minimum area threshold
            area = int(np.sum(binary_mask > 0))
            if area < min_area_px:
                continue

            # Find largest contour outlining the detection
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                # Simplify polygon contour using Douglas-Peucker algorithm
                epsilon = 0.02 * cv2.arcLength(largest, True)
                polygon = cv2.approxPolyDP(largest, epsilon, True)

            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": conf,
                "bbox": bbox,
                "mask": binary_mask,
                "area_px": area,
                "polygon": polygon,
                "track_id": track_id
            })

        return detections
