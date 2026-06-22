<div align="center">

# 🛢️ **Spill & Leak Detection Engine**

### 🚨 Real-Time Oil & Chemical Leak Detection for Industrial CCTV Systems

A **production-grade AI pipeline** built for **real-time CCTV / RTSP monitoring**, combining **fast YOLO segmentation + custom tracking** for high-precision alerts.

> ⚙️ Powered by **YOLOv8 TensorRT (Segmentation & Tracking)**
> 🧠 Designed for **low false positives, high reliability deployments**
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![YOLO](https://img.shields.io/badge/YOLO-TensorRT-success?logo=nvidia&logoColor=white)](#)
[![GStreamer](https://img.shields.io/badge/GStreamer-H264/H265-blue)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## Project Structure
```
├── config/
│   ├── config.yaml          # Unified model, camera, and output settings
│   └── roi.yaml             # Cached Region of Interest (ROI) polygon boundaries
│
├── models/
│   └── oil_spill_seg.engine # GPU-accelerated TensorRT engine weights
│
├── src/
│   ├── camera/
│   │   └── rtsp_reader.py   # Threaded camera frame reader supporting standard & GStreamer streams
│   ├── detector/
│   │   └── spill_detector.py# YOLO tracking and segmentation wrapper
│   ├── roi/
│   │   └── roi_manager.py   # Mouse-click based manual ROI selector and filter
│   ├── visualization/
│   │   └── draw_results.py  # Annotation and overlay drawing
│   ├── output/
│   │   ├── save_image.py    # Image screenshot saver (1 per unique track ID)
│   │   ├── save_video.py    # Custom duration video clip recorder on detection
│   │   └── save_json.py     # JSON logs history writer
│   └── utils/
│       ├── logger.py        # Rotating log recorder
│       └── helpers.py       # YAML parser utilities
│
├── outputs/
│   ├── images/              # Custom folder path for screenshots (default)
│   ├── videos/              # Custom folder path for video clips (default)
│   └── json/                # Custom folder path for JSON logs (default)
│
├── logs/                    # Rotating log files
├── main.py                  # Main entry point
└── requirements.txt         # Project dependencies
```

## Quick Start

### 1. Configure the System
Open `config/config.yaml` to specify:
- **`camera.mode`**: Set to `"stream"` (for live streams/videos) or `"image"` (for single static image tests).
- **`camera.source`**: RTSP URL, local video file path, webcam index, or local image file path.
- **`model.weights`**: TensorRT engine weights path (defaults to `models/oil_spill_seg.engine`).
- Custom output paths for saved images, videos, and JSON reports.
- Video clip duration (seconds).

### 2. Start the Pipeline
Activate your conda environment and run:
```bash
conda run -n spill python main.py
```

### 3. Draw or Update your ROI
- **On Startup**: A GUI window will open showing the first frame. Left-click points to draw your Region of Interest (ROI) polygon. Press **ENTER** (or Space) to save it and begin processing. Press **ESC** (or Q) to skip ROI filtering (monitors the entire frame).
- **Caching**: The ROI is automatically saved to `config/roi.yaml`. On startup, you can press **ENTER** immediately to reuse the cached polygon.
- **On-the-fly Resetting**: (Stream mode only) Press **`R`** on the live preview window at any time while the application is running to pause execution and draw a new ROI on the current frame.

---

## Features
- **Flexible Inputs**: Easily switch between static images and video feeds in `config.yaml`.
- **GPU Acceleration**: Built-in support for YOLO TensorRT `.engine` models.
- **Hardware-Accelerated Decoding**: Automatically activates the GStreamer backend (`cv2.CAP_GSTREAMER`) if camera source contains a pipeline string (indicated by the `!` character).
- **Custom Directories**: Configurable output locations in `config/config.yaml`.
- **Redundancy Filter**: Saves exactly **one** screenshot per unique leak tracking ID.
- **Custom Event Recording**: When a spill is detected inside your ROI, a video clip of your configured length (e.g. 5 seconds) is recorded showing all bounding boxes and masks.
- **Auto-Reconnection**: The RTSP reader automatically reconnects if the camera connection drops.
