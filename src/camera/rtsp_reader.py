import cv2
import threading
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RTSPReader:
    """Threaded RTSP / Camera frame reader supporting standard feeds, USB webcams, and GStreamer pipelines."""
    def __init__(self, source, width=None, height=None, fps_limit=None):
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height
        self.fps_limit = fps_limit

        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        self._connect()

    def _connect(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Connecting to camera source: {self.source}")
        
        # Check source type
        is_gstreamer = isinstance(self.source, str) and '!' in self.source
        is_webcam = isinstance(self.source, int)

        if is_gstreamer:
            logger.info("Initializing OpenCV VideoCapture with GStreamer backend (cv2.CAP_GSTREAMER).")
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        elif is_webcam:
            logger.info(f"Initializing USB webcam index {self.source} with V4L2 backend.")
            # V4L2 (Video4Linux2) is the standard and most reliable backend for USB webcams on Linux
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                logger.warning("V4L2 backend failed to open webcam, retrying with default backend.")
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera source: {self.source}")
            return False

        # Apply resolution overrides (only if not using a GStreamer pipeline)
        if not is_gstreamer:
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        logger.info("Camera connection established successfully.")
        return True

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("RTSPReader thread started.")

    def _update(self):
        sleep_time = 1.0 / self.fps_limit if self.fps_limit and self.fps_limit > 0 else 0.001
        consecutive_failures = 0

        while self.running:
            t_start = time.time()

            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera connection lost. Reconnecting...")
                if self._connect():
                    consecutive_failures = 0
                else:
                    time.sleep(2.0)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame from camera (failure count: {consecutive_failures}).")
                with self.lock:
                    self.ret = False
                
                # Reconnect if we fail repeatedly
                if consecutive_failures >= 5:
                    logger.error("Too many consecutive read failures. Attempting reconnection...")
                    self._connect()
                    consecutive_failures = 0
                
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            with self.lock:
                self.ret = True
                self.frame = frame

            elapsed = time.time() - t_start
            delay = sleep_time - elapsed
            if delay > 0:
                time.sleep(delay)

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("RTSPReader released.")
