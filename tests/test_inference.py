import numpy as np
import pytest
from src.detector.spill_detector import SpillDetector
from src.roi.roi_manager import ROIManager
from src.visualization.draw_results import draw_results

def test_detector_initialization():
    # Test that the detector can be initialized
    detector = SpillDetector(
        weights_path="models/best.pt",
        device="cpu",  # Use CPU for tests to ensure portability
        imgsz=640,
        conf=0.45,
        iou=0.45
    )
    assert detector is not None
    assert detector.device == "cpu"

def test_roi_manager_filtering():
    # Test ROI manager initialization and filtering
    roi_manager = ROIManager("config/roi.yaml")
    assert roi_manager is not None
    
    # Setup test-specific ROIs to ensure isolation from the file on disk
    roi_manager.rois = [{
        "name": "zone_1",
        "polygon": np.array([[100, 100], [1180, 100], [1180, 620], [100, 620]], dtype=np.int32)
    }]
    
    # Setup mock detections
    detections = [
        {
            "class_id": 0,
            "class_name": "oil",
            "confidence": 0.9,
            "bbox": [150, 150, 200, 200],
            "polygon": np.array([[150, 150], [200, 150], [200, 200], [150, 200]], dtype=np.int32)
        },
        {
            "class_id": 0,
            "class_name": "oil",
            "confidence": 0.9,
            "bbox": [5, 5, 20, 20],
            "polygon": np.array([[5, 5], [20, 5], [20, 20], [5, 20]], dtype=np.int32)
        }
    ]
    
    filtered = roi_manager.filter_detections(detections)
    # The second detection (5, 5) is outside the zone_1 polygon ([100, 100] to [1180, 620])
    # The first detection (150, 150) is inside it
    assert len(filtered) == 1
    assert filtered[0]["bbox"] == [150, 150, 200, 200]
    assert filtered[0]["roi_name"] == "zone_1"
