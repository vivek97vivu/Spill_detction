import cv2
import time
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ImageSaver:
    """Handles saving frame images (e.g. when a spill is detected)."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frame, name_prefix: str = "detection") -> Path:
        """Saves a frame to the outputs directory with a millisecond timestamp."""
        timestamp = int(time.time() * 1000)
        filename = self.output_dir / f"{name_prefix}_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved image: {filename.absolute()}")
        return filename
