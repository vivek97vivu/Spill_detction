import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating File handler (max 10MB per file, keeping up to 5 files)
    fh = RotatingFileHandler(
        log_dir / "spill_detection.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
