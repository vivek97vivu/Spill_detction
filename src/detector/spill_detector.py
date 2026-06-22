import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SpillDetector:
    """Wrapper for YOLO segmentation and tracking model to detect spills and track instances."""
    def __init__(self, weights_path: str, person_weights_path: str = None, device: str = "cuda", imgsz: int = 640, conf: float = 0.45, iou: float = 0.45):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.weights_path.absolute()}")

        logger.info(f"Loading YOLO spill model from {self.weights_path} on device {device}")
        self.model = YOLO(str(self.weights_path))
        if self.weights_path.suffix == ".pt":
            self.model.to(device)

        self.person_model = None
        if person_weights_path:
            p_path = Path(person_weights_path)
            if p_path.exists():
                logger.info(f"Loading YOLO person model from {p_path} on device {device}")
                self.person_model = YOLO(str(p_path))
                if p_path.suffix == ".pt":
                    self.person_model.to(device)
            else:
                logger.warning(f"Person model file not found at: {p_path.absolute()}. Person bypass will be disabled.")

        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        # Warmup models
        logger.info("Warming up models...")
        dummy = np.zeros((imgsz, imgsz, 3), dtype="uint8")
        self.predict(dummy)
        if self.person_model is not None:
            self.person_model.predict(source=dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        logger.info("Warmup complete.")

    def predict(self, frame: np.ndarray):
        """Perform raw YOLO spill model inference (used for warmup)."""
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
        """Perform YOLO spill model tracking on the frame sequence."""
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
        
        Runs the person model to find people and the spill model to find spills.
        Filters out any spill detections that overlap significantly with detected persons
        to prevent false positive spill alerts on people.
        """
        detections = []
        h, w = frame.shape[:2]

        # 1. Run Person Detection (yolo26s.pt)
        person_boxes = []
        if self.person_model is not None:
            person_results = self.person_model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )[0]
            
            if person_results.boxes is not None:
                for box in person_results.boxes:
                    cls_id = int(box.cls[0])
                    # In COCO models (yolo26s.pt), class 0 is 'person'
                    if cls_id == 0:
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        person_boxes.append(bbox)
                        
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0].item())
                            
                        x1, y1, x2, y2 = bbox
                        area = int((x2 - x1) * (y2 - y1))
                        polygon = np.array([
                            [[int(x1), int(y1)]],
                            [[int(x2), int(y1)]],
                            [[int(x2), int(y2)]],
                            [[int(x1), int(y2)]]
                        ], dtype=np.int32)
                        
                        detections.append({
                            "class_id": 0,
                            "class_name": "Person",
                            "confidence": conf,
                            "bbox": bbox,
                            "mask": None,
                            "area_px": area,
                            "polygon": polygon,
                            "track_id": track_id
                        })

        # 2. Run Spill Detection (best.pt)
        result = self.track(frame, persist=True)
        if result.boxes is None:
            return detections
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            
            # Since best.pt might also output Person (0), we only want Spill (1) detections
            if name != "Spill" and cls_id != 1:
                continue

            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            # Check if this spill box overlaps with any detected person to filter out false positives
            is_false_positive = False
            for p_box in person_boxes:
                # Calculate intersection over spill box area
                x1 = max(bbox[0], p_box[0])
                y1 = max(bbox[1], p_box[1])
                x2 = min(bbox[2], p_box[2])
                y2 = min(bbox[3], p_box[3])
                
                if x2 > x1 and y2 > y1:
                    inter_area = (x2 - x1) * (y2 - y1)
                    spill_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if spill_area > 0:
                        overlap_ratio = inter_area / spill_area
                        if overlap_ratio > 0.4:  # 40% overlap threshold
                            is_false_positive = True
                            break

            if is_false_positive:
                logger.info("Filtered out a false positive spill detection overlapping with a person.")
                continue

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            binary_mask = None
            area = 0
            polygon = None

            if result.masks is not None:
                mask_data = result.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask_data.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask_resized > 0.5).astype("uint8") * 255
                area = int(np.sum(binary_mask > 0))

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    epsilon = 0.02 * cv2.arcLength(largest, True)
                    polygon = cv2.approxPolyDP(largest, epsilon, True)
            else:
                x1, y1, x2, y2 = bbox
                area = int((x2 - x1) * (y2 - y1))
                polygon = np.array([
                    [[int(x1), int(y1)]],
                    [[int(x2), int(y1)]],
                    [[int(x2), int(y2)]],
                    [[int(x1), int(y2)]]
                ], dtype=np.int32)

            if area < min_area_px:
                continue

            detections.append({
                "class_id": cls_id,
                "class_name": "Spill",
                "confidence": conf,
                "bbox": bbox,
                "mask": binary_mask,
                "area_px": area,
                "polygon": polygon,
                "track_id": track_id
            })

        return detections



