import json
import time
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class JSONSaver:
    """Logs structured spill detection metadata and saves to JSON files."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.session_id = int(time.time())

    def log_detections(self, detections: list[dict]):
        """Logs active detections to history with metadata."""
        timestamp = time.time()
        for det in detections:
            self.history.append({
                "timestamp": timestamp,
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "bbox": [round(val, 1) for val in det["bbox"]],
                "area_px": det.get("area_px"),
                "track_id": det.get("track_id"),
                "roi_name": det.get("roi_name", "")
            })

    def save(self) -> Path | None:
        """Writes logged history to a JSON report file."""
        if not self.history:
            logger.info("No detections recorded; skipping JSON report generation.")
            return None

        filepath = self.output_dir / f"report_{self.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved JSON report: {filepath.absolute()}")
        return filepath

    def save_individual_report(self, detections: list[dict], filename_prefix: str, timestamp_ms: int) -> Path:
        """Saves details of a single snapshot's detections to an individual JSON file."""
        data = []
        for det in detections:
            data.append({
                "timestamp_ms": timestamp_ms,
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "bbox": [round(val, 1) for val in det["bbox"]],
                "area_px": det.get("area_px"),
                "track_id": det.get("track_id"),
                "roi_name": det.get("roi_name", "")
            })
            
        filepath = self.output_dir / f"{filename_prefix}_{timestamp_ms}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved individual JSON report: {filepath.absolute()}")
        return filepath

