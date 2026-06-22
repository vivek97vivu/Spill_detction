import cv2
import time
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VideoSaver:
    """Handles continuous video recording of detection sessions."""
    def __init__(self, output_dir: str, fps: int = 10, width: int = 1280, height: int = 720, codec: str = "mp4v"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.writer = None
        self.filepath = None

    def start(self):
        """Initializes VideoWriter with dynamic timestamped filename."""
        timestamp = int(time.time())
        self.filepath = self.output_dir / f"session_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.filepath),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        logger.info(f"Started video recording: {self.filepath.absolute()}")

    def write(self, frame):
        """Writes a frame to the video, resizing if necessary."""
        if self.writer is None:
            self.start()

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        self.writer.write(frame)

    def release(self):
        """Releases the VideoWriter resource."""
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info(f"Video saved: {self.filepath.absolute()}")
