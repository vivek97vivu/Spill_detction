import cv2
import time
import sys
from pathlib import Path

from src.utils.helpers import load_yaml
from src.utils.logger import get_logger
from src.camera.rtsp_reader import RTSPReader
from src.detector.spill_detector import SpillDetector
from src.roi.roi_manager import ROIManager
from src.visualization.draw_results import draw_results
from src.output.save_image import ImageSaver
from src.output.save_video import VideoSaver
from src.output.save_json import JSONSaver

logger = get_logger("main")

def main():
    logger.info("Initializing Oil Leak Detection System...")

    # 1. Load configuration file
    try:
        config = load_yaml("config/config.yaml")
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)

    # 2. Extract parameters
    cam_settings = config["camera"]
    mode = cam_settings.get("mode", "stream").lower()
    
    model_settings = config["model"]
    class_names = {int(k): v for k, v in model_settings["classes"].items()}
    color_map = config["visualization"]["colors"]
    mask_alpha = config["visualization"].get("mask_alpha", 0.4)
    show_preview = config["visualization"].get("show_preview", True)
    preview_scale = config["visualization"].get("preview_scale", 1.0)
    min_area_px = config["postprocess"].get("min_area_px", 500)

    # Output Savers Configurations
    output_cfg = config.get("output", {})
    image_dir = output_cfg.get("image_dir", "outputs/images")
    video_dir = output_cfg.get("video_dir", "outputs/videos")
    json_dir = output_cfg.get("json_dir", "outputs/json")
    video_duration_sec = output_cfg.get("video_duration_sec", 5)

    # Instantiate detector and savers
    try:
        # Spill Detector
        detector = SpillDetector(
            weights_path=model_settings["weights"],
            person_weights_path=model_settings.get("person_weights"),
            device=model_settings["device"],
            imgsz=model_settings.get("imgsz", 640),
            conf=model_settings.get("conf", 0.45),
            iou=model_settings.get("iou", 0.45)
        )

        # Region of Interest (ROI) Manager
        roi_manager = ROIManager("config/roi.yaml")

        # Savers
        image_saver = ImageSaver(image_dir)
        json_saver = JSONSaver(json_dir)

    except Exception as e:
        logger.exception(f"Initialization of core components failed: {e}")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────
    # SINGLE IMAGE MODE
    # ─────────────────────────────────────────────────────────────
    if mode == "image":
        logger.info("Running in Single Image Mode.")
        image_source = cam_settings["source"]
        
        # Load static image
        frame = cv2.imread(image_source)
        if frame is None:
            logger.error(f"Failed to read image source: {image_source}. Make sure file path is correct.")
            sys.exit(1)

        logger.info(f"Loaded image: {image_source}. Opening ROI selection window...")
        roi_manager.select_roi_interactively(frame)

        # Spill segmentation
        raw_detections = detector.detect_spills(
            frame,
            min_area_px=min_area_px,
            class_names=class_names
        )

        # ROI filtering
        detections = roi_manager.filter_detections(raw_detections)

        # Render overlays
        annotated_frame = draw_results(
            frame,
            detections,
            roi_manager=roi_manager,
            color_map=color_map,
            mask_alpha=mask_alpha
        )

        # Save image and JSON log only if a spill is detected
        spill_detections = [d for d in detections if d.get("class_name") == "Spill"]
        if len(spill_detections) > 0:
            timestamp_ms = int(time.time() * 1000)
            image_saver.save(annotated_frame, name_prefix="detection_img", timestamp=timestamp_ms)
            json_saver.save_individual_report(spill_detections, filename_prefix="detection_img", timestamp_ms=timestamp_ms)
            # Log to session tracker and save report
            json_saver.log_detections(spill_detections)
            json_saver.save()
            logger.info("Spill detection snapshot and JSON saved to outputs.")
        else:
            logger.info("No spills detected inside the ROI.")

        # Show preview window
        if show_preview:
            try:
                h, w = annotated_frame.shape[:2]
                w_scaled = int(w * preview_scale)
                h_scaled = int(h * preview_scale)
                preview = cv2.resize(annotated_frame, (w_scaled, h_scaled))
                
                logger.info("Displaying annotated image preview. Press any key in the window to exit.")
                cv2.imshow("Oil Leak Detection - Single Image", preview)
                cv2.waitKey(0)
            except cv2.error as e:
                logger.warning(f"Failed to display GUI window (preview unavailable): {e}")

        cv2.destroyAllWindows()
        logger.info("Single image processing complete.")
        return

    # ─────────────────────────────────────────────────────────────
    # REAL-TIME STREAM MODE (Webcam / Video file / RTSP Feed)
    # ─────────────────────────────────────────────────────────────
    elif mode == "stream":
        logger.info("Running in Stream Mode (Live RTSP / Video file).")
        fps = cam_settings.get("fps_limit", 10)
        
        try:
            reader = RTSPReader(
                source=cam_settings["source"],
                width=cam_settings.get("width"),
                height=cam_settings.get("height"),
                fps_limit=fps
            )
        except Exception as e:
            logger.exception(f"Failed to initialize Camera Reader: {e}")
            sys.exit(1)

        # Start threaded reader
        reader.start()
        logger.info("Waiting for first camera frame...")
        
        first_frame = None
        for _ in range(50):
            ret, frame = reader.read()
            if ret and frame is not None:
                first_frame = frame
                break
            time.sleep(0.1)

        if first_frame is not None:
            logger.info("First frame captured successfully. Starting ROI selection...")
            roi_manager.select_roi_interactively(first_frame)
        else:
            logger.error("Failed to capture first frame for ROI selection. Skipping ROI selection.")

        processed_track_ids = set()
        active_videos = {}
        untracked_counter = 90000

        logger.info("Oil Leak Detection pipeline running. Controls in preview window:")
        logger.info(" - 'q' key: Exit application")
        logger.info(" - 'r' key: Redraw Region of Interest (ROI) on-the-fly")

        try:
            while True:
                ret, frame = reader.read()
                if not ret or frame is None:
                    # Sleep briefly if reader doesn't have a new frame yet
                    time.sleep(0.01)
                    continue

                # Spill segmentation and tracking
                raw_detections = detector.detect_spills(
                    frame,
                    min_area_px=min_area_px,
                    class_names=class_names
                )

                # Region of Interest (ROI) filtering
                detections = roi_manager.filter_detections(raw_detections)

                # Render overlays
                annotated_frame = draw_results(
                    frame,
                    detections,
                    roi_manager=roi_manager,
                    color_map=color_map,
                    mask_alpha=mask_alpha
                )

                # Handle detection actions (alerts and recording trigger only for Spills)
                spill_detections = [d for d in detections if d.get("class_name") == "Spill"]
                for det in spill_detections:
                    track_id = det.get("track_id")
                    if track_id is None:
                        # Fallback for untracked spills to ensure they have a unique track ID
                        track_id = f"untracked_{untracked_counter}"
                        untracked_counter += 1
                        det["track_id"] = track_id

                    if track_id not in processed_track_ids:
                        processed_track_ids.add(track_id)
                        logger.info(f"New spill detected! Track ID: {track_id}")

                        # 1. Save one image showing the first appearance of the spill
                        timestamp_ms = int(time.time() * 1000)
                        image_name_prefix = f"spill_track_{track_id}"
                        image_saver.save(annotated_frame, name_prefix=image_name_prefix, timestamp=timestamp_ms)

                        # 2. Save one JSON report
                        json_saver.save_individual_report([det], filename_prefix=image_name_prefix, timestamp_ms=timestamp_ms)

                        # Also log to session history
                        json_saver.log_detections([det])

                        # 3. Start recording a video clip for that spill event
                        h, w = frame.shape[:2]
                        vid_saver = VideoSaver(
                            output_dir=video_dir,
                            fps=fps,
                            width=w,
                            height=h
                        )
                        vid_saver.start(filename_prefix=image_name_prefix)
                        active_videos[track_id] = {
                            "saver": vid_saver,
                            "frames_left": int(video_duration_sec * fps)
                        }

                # Write frame to all active video recordings
                if active_videos:
                    finished_tracks = []
                    for t_id, video_info in active_videos.items():
                        video_info["saver"].write(annotated_frame)
                        video_info["frames_left"] -= 1
                        if video_info["frames_left"] <= 0:
                            video_info["saver"].release()
                            finished_tracks.append(t_id)
                            logger.info(f"Finished recording {video_duration_sec}-second video clip for track ID {t_id}.")
                    
                    for t_id in finished_tracks:
                        active_videos.pop(t_id)

                # Show live preview
                if show_preview:
                    try:
                        h, w = annotated_frame.shape[:2]
                        w_scaled = int(w * preview_scale)
                        h_scaled = int(h * preview_scale)
                        preview = cv2.resize(annotated_frame, (w_scaled, h_scaled))
                        
                        cv2.imshow("Oil Leak Detection Preview", preview)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q') or key == ord('Q'):
                            logger.info("User requested exit.")
                            break
                        elif key == ord('r') or key == ord('R'):
                            logger.info("Reset ROI requested. Opening interactive drawing window...")
                            success, current_frame = reader.read()
                            if success and current_frame is not None:
                                cv2.destroyWindow("Oil Leak Detection Preview")
                                roi_manager.select_roi_interactively(current_frame)
                                processed_track_ids.clear()
                            else:
                                logger.error("Failed to capture latest frame for ROI selection.")
                                
                    except cv2.error as e:
                        logger.warning(f"Failed to display GUI window (disabled show_preview): {e}")
                        show_preview = False

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        except Exception as e:
            logger.exception(f"Unhandled exception in pipeline: {e}")
        finally:
            logger.info("Stopping and cleaning up components...")
            reader.release()
            
            for t_id, video_info in active_videos.items():
                video_info["saver"].release()
            active_videos.clear()
                
            json_saver.save()
            cv2.destroyAllWindows()
            logger.info("Shutdown process complete.")
            
    else:
        logger.error(f"Invalid camera mode configured: '{mode}'. Must be 'stream' or 'image'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
