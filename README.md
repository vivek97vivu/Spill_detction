<div align="center">

# 🛢️ **Spill & Leak Detection Engine (RF-DETR)**

### 🚨 Real-Time Oil & Chemical Leak Detection for Industrial CCTV Systems

A **production-grade AI pipeline** built for **real-time CCTV / RTSP monitoring**, combining **RF-DETR Object Detection + ByteTrack tracking + interactive Region of Interest (ROI) filtering** for high-precision alerts.

> ⚙️ Powered by **Roboflow RF-DETR (Real-Time Detection Transformer)**
> 🧠 Integrated with **ByteTrack** for high-accuracy bounding box tracking
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![RF-DETR](https://img.shields.io/badge/RF--DETR-PyTorch-orange?logo=pytorch&logoColor=white)](#)
[![GStreamer](https://img.shields.io/badge/GStreamer-H264/H265-blue)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## ⚡ Core Stack

| Component | Purpose |
|---|---|
| 🛢️ **RF-DETR Detection** | Real-time leak bounding box detection using Roboflow Detection Transformer (`Spill`) |
| 🛡️ **Interactive ROI Manager** | Mouse-click polygon drawing window to isolate detection zones |
| 🎥 **Threaded RTSP Reader** | Low-latency H264/H265 hardware-accelerated GStreamer & RTSP reader |
| 🔁 **ByteTrack Multi-Object Tracking** | Associates bounding boxes across frames to assign unique Track IDs |
| 📝 **JSON Logger** | Auto-saving structured records of detections, track IDs, and pixel area metrics |
| ⚙️ **YAML Config Engine** | Centrally managed settings for camera inputs, thresholds, and outputs |

---

## 🚀 Pipeline Overview

```text
Camera (RTSP Stream / Webcam / Video File / Static Image)
                          ↓
        Threaded Frame Reader (RTSPReader)
                          ↓
    Interactive ROI Polygon Selection on Startup
                          ↓
             RF-DETR Detection Model
                          ↓
             ByteTrack Object Tracking
                          ↓
     Contour Area & ROI Polygon Collision Filtering
                          ↓
       Track ID Deduplication (1 Alert per Spill ID)
                          ↓
  🚨 ALERT + Save (Annotated Image, Video Clip, JSON Log)
```

---

## 🎯 Key Features

* 🛢️ **Real-Time Spill Detection**: Direct integration with the state-of-the-art RF-DETR transformer-based detector for fast and accurate leak localization.
* 🛡️ **Interactive ROI Selector**: Graphically draw polygon boundaries on startup to filter out irrelevant areas (caching coordinates to `config/roi.yaml`).
* 🔄 **On-the-Fly ROI Resetting**: Press `R` on the live window to redraw the ROI on the latest frame without restarting the pipeline.
* ⚡ **ByteTrack Tracking**: Assigns unique track IDs to spills, ensuring consistent tracking across frame sequences.
* 🔁 **Redundancy Filter**: Prevents alert flooding by saving exactly one screenshot, one JSON report, and one video clip per unique track ID.
* 🎥 **Custom Track Recording**: Records a track-specific video clip of configurable length (e.g., 5s) showing bounding boxes when a new spill is detected in the ROI.
* 📂 **Structured JSON Logging**: Saves individual JSON reports and session history tracking timestamps, areas, and confidence metadata for audit trails.
* 🖥️ **Headless GUI Fallback**: Robustly catches display/GUI exceptions on server environments to reuse previously configured ROIs instead of crashing.
* 🏷️ **Track ID Annotation**: Displays track IDs (e.g., `Spill #5`) directly on the live overlay and saved screenshots for easy auditing.

---

## 📂 Project Structure

```bash
spills/
├── config/
│   ├── config.yaml             # Main configuration file (model, camera, output parameters)
│   └── roi.yaml                # Cached ROI coordinates
│
├── models/
│   └── checkpoint_best_ema(2).pth  # RF-DETR model checkpoint file
│
├── outputs/
│   ├── images/                 # Confirmed spill screenshots
│   ├── json/                   # Audit log files
│   └── videos/                 # Confirmed spill video recordings
│
├── src/
│   ├── camera/
│   │   └── rtsp_reader.py      # Threaded frame reader (supporting standard/GStreamer)
│   ├── detector/
│   │   └── spill_detector.py   # Model wrapper supporting both YOLO and RF-DETR
│   ├── roi/
│   │   └── roi_manager.py      # GUI-based manual ROI selector & geometry filter
│   ├── visualization/
│   │   └── draw_results.py     # Renders bbox overlays and status indicators
│   ├── output/
│   │   ├── save_image.py       # Alert screenshot saver (1 per unique ID)
│   │   ├── save_video.py       # Temporary clip writer for detection events
│   │   └── save_json.py        # Periodic JSON history logger
│   └── utils/
│       ├── logger.py           # Rotating log configuration
│       └── helpers.py          # YAML loading helpers
│
├── main.py                     # Main application entry point
└── README.md
```

---

## ⚙️ Configuration

All system behavior is controlled via `config/config.yaml`. No code changes needed.

```yaml
model:
  weights: "models/checkpoint_best_ema(2).pth" 
  person_weights: null                   
  device: "cuda"                         
  imgsz: 640                           
  conf: 0.45                        
  iou: 0.45                            
  classes:
    0: Spill

postprocess:
  min_area_px: 500

visualization:
  show_preview: true                   
  preview_scale: 0.75
  mask_alpha: 0.4
  colors:
    Spill: [255, 0, 0]                   

camera:
  mode: "stream"  
  source: "1.mp4" 
  fps_limit: 25
  width: 1280
  height: 720

output:
  image_dir: "outputs/images"         
  video_dir: "outputs/videos"          
  json_dir: "outputs/json"             
  video_duration_sec: 5
```

---

## 🚀 Installation

```bash
git clone https://github.com/vivek97vivu/Spill_detction.git
cd Spill_detction

# Activate your conda environment (e.g., spill)
conda activate spill

# Install required dependencies
pip install -r requirements.txt
pip install rfdetr
```

### Requirements

* NVIDIA GPU (CUDA support recommended)
* CUDA 12.x / 13.x
* OpenCV with GStreamer support (optional, for low-latency RTSP pipelines)
* Python 3.10+
* `rfdetr` package and its dependencies (`supervision`, `transformers`, etc.)

---

## ▶️ Run

```bash
python main.py
```

---

## 🎥 Controls

| Key | Action |
|---|---|
| `q` / `Q` / `ESC` | Quit the application |
| `r` / `R` | Reset and redraw the Region of Interest (ROI) on-the-fly |
| `ENTER` / `SPACE` | Save ROI and start processing (during interactive ROI setup) |

---

## 🚨 Alert System

### Stage 1 — Detection & Tracking
RF-DETR detects spill bounding boxes, and ByteTrack maintains consistent tracking IDs across frame sequences.

### Stage 2 — ROI Intersection
If the detected spill box intersects with your custom ROI boundary and exceeds `min_area_px`:
* **Image Alert**: Saves a high-quality annotated JPEG showing the first appearance of the spill to `outputs/images/` (exactly once per track ID, named `spill_track_{track_id}_{timestamp}.jpg`).
* **Video Alert**: Starts recording a `video_duration_sec` track-specific video clip showing the incident to `outputs/videos/` (exactly once per track ID, named `spill_track_{track_id}_{timestamp}.mp4`).
* **JSON Log**: Saves an individual JSON report detailing the incident to `outputs/json/` (exactly once per track ID, named `spill_track_{track_id}_{timestamp}.json`).

---

## 📸 Output

| Folder | Contents |
|---|---|
| `outputs/images/` | Detection screenshots (saved exactly once per track ID) |
| `outputs/videos/` | Event-triggered video clips containing overlays |
| `outputs/json/` | Serialized detection logs containing pixel area, track IDs, and timestamps |

---

## ⚡ Performance

| Metric | Value |
|---|---|
| RF-DETR Inference Latency | ~16–18ms (on RTX 5060 GPU with optimization) |
| Post-processing & ROI Check | ~1–2ms |
| RTSP Read Latency | Thread-separated, frame-drop policy (0ms block on main loop) |
| GPU Memory Footprint | ~1.5 GB to 2.5 GB (depending on resolution) |

---

## 🧪 Engineering Decisions

| Decision | Reason |
|---|---|
| **Interactive ROI Selection** | CCTV streams contain static areas (buildings, skies) that never leak; ROI limits false alerts. |
| **ByteTrack Tracking** | Leverages Supervision's high-performance tracker to ensure that multiple spills are cleanly separated and unique alerts are sent. |
| **Inference Optimization** | Compiles the PyTorch transformer graph using TorchScript trace for faster inference on the GPU. |
| **Per-ID Deduplication** | Keeps alert directories clean by avoiding flood-saving the same leak every frame, enforcing exactly one image, one JSON report, and one video clip per spill track ID. |
| **Threaded RTSP Reader** | Runs frame reading in a background thread to prevent GUI lagging and buffer accumulation. |
| **JSON Logger** | Keeps a machine-readable audit trail of leaks, track IDs, and pixel areas. |
| **On-the-Fly ROI Reset** | Allows security operators to adjust monitoring boundaries live without restarting the pipeline. |
| **Headless GUI Fallback** | Prevents pipeline crashes in remote or automated headless server runs by gracefully catching GUI display errors and reusing loaded config. |

---

## 🔮 Future Enhancements

* 📧 Email/Slack notification integration on detection events
* ☁️ Cloud storage synchronization (AWS S3) for alerts
* 🌐 Web dashboard for multi-camera visualization
* 📊 Real-time leakage area growth-rate graph visualization

---

<div align="center">
Engineered by the <b>Algosium AI Team</b> · CampNeuron AI Series
</div>
