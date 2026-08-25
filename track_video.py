import argparse
import csv
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

from detect_video import discover_avenue_videos


def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """Generate a consistent, visually distinct BGR color for a given track ID."""
    np.random.seed(int(track_id) * 37)
    color = np.random.randint(50, 240, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_track_annotation(
    frame: np.ndarray,
    track_id: int,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    foot_x: float,
    foot_y: float,
) -> None:
    """Draw bounding box, track ID label, confidence, and foot location marker on a frame."""
    color = get_color_for_id(track_id)
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    ifoot_x, ifoot_y = int(round(foot_x)), int(round(foot_y))

    # 1. Bounding box
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

    # 2. Foot location marker
    cv2.circle(frame, (ifoot_x, ifoot_y), 4, (0, 0, 255), -1)

    # 3. Label (ID + Confidence)
    label = f"ID {track_id} ({conf:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y1 = max(0, iy1 - text_h - 6)
    label_y2 = iy1
    label_x1 = ix1
    label_x2 = ix1 + text_w + 6

    # Background rectangle for text readability
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (label_x1 + 3, label_y2 - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def track_video(
    video_path: str | Path | None = None,
    model_path: str | Path = "yolov8x.pt",
    output_dir: str | Path = "outputs/tracking",
    output_video_name: str = "avenue_tracking.mp4",
    output_csv_name: str = "avenue_tracking.csv",
    tracker_config: str = "bytetrack.yaml",
    conf: float = 0.15,
    imgsz: int = 640,
    max_frames: int | None = None,
) -> dict:
    """Execute YOLOv8 person detection with ByteTrack persistent ID tracking on a CCTV video.

    Args:
        video_path: Path to input video. If None, defaults to first Avenue training video.
        model_path: Path to YOLOv8 weights.
        output_dir: Directory to save tracking video and CSV records.
        output_video_name: Filename of the output tracking video.
        output_csv_name: Filename of the tracking CSV file.
        tracker_config: Tracking configuration file (default: 'bytetrack.yaml').
        conf: Confidence threshold for person detection.
        imgsz: Inference frame size.
        max_frames: Maximum number of frames to process (None for full video).

    Returns:
        Dictionary containing tracking metrics and output file locations.
    """
    # 1. Resolve input video path
    if video_path is None:
        discovered = discover_avenue_videos()
        if not discovered:
            raise FileNotFoundError(
                "No Avenue training videos found in 'datasets/Avenue_Dataset/Avenue Dataset/training_videos'."
            )
        input_file = discovered[0]
    else:
        input_file = Path(video_path)
        if not input_file.exists():
            fallback = Path("datasets/Avenue_Dataset/Avenue Dataset/training_videos") / input_file.name
            if fallback.exists():
                input_file = fallback
            else:
                raise FileNotFoundError(f"Input video file not found: {video_path}")

    # 2. Setup output paths
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = out_dir / output_video_name
    output_csv_path = out_dir / output_csv_name

    # 3. Open input video stream
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_file}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Initialize Windows-compatible VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        output_video_path = out_dir / f"{output_video_path.stem}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer for: {output_video_path}")

    # 5. Open CSV writer
    csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
        "foot_x",
        "foot_y",
        "confidence",
    ]
    csv_writer.writerow(csv_header)

    # 6. Load YOLOv8 Model
    model = YOLO(str(model_path))

    # 7. Frame-by-frame tracking loop
    processed_frames = 0
    total_tracking_records = 0
    unique_track_ids = set()
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame_idx = processed_frames + 1

            # Run YOLO + ByteTrack
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_config,
                classes=[0],
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # Check if detections with track IDs exist in the frame
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for tid, conf_val, (x1, y1, x2, y2) in zip(track_ids, confs, xyxys):
                    # Compute bounding box and ground contact foot point metrics
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    # Write tracking record to CSV
                    csv_writer.writerow([
                        current_frame_idx,
                        tid,
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        f"{bbox_w:.2f}",
                        f"{bbox_h:.2f}",
                        f"{center_x:.2f}",
                        f"{center_y:.2f}",
                        f"{foot_x:.2f}",
                        f"{foot_y:.2f}",
                        f"{conf_val:.4f}",
                    ])

                    unique_track_ids.add(tid)
                    total_tracking_records += 1

                    # Draw on frame
                    draw_track_annotation(
                        frame=frame,
                        track_id=tid,
                        conf=conf_val,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        foot_x=foot_x,
                        foot_y=foot_y,
                    )

            # Write frame to video stream
            writer.write(frame)
            processed_frames += 1

            if max_frames is not None and processed_frames >= max_frames:
                break

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

    # 8. Output Summary
    print("=" * 60)
    print("Avenue CCTV Video Tracking Summary (YOLOv8 + ByteTrack)")
    print("=" * 60)
    print(f"input video:              {input_file}")
    print(f"output tracking video:    {output_video_path}")
    print(f"total frames processed:   {processed_frames}")
    print(f"total unique track IDs:   {len(unique_track_ids)}")
    print(f"total tracking records:   {total_tracking_records}")
    print(f"average processing FPS:   {avg_fps:.2f}")
    print(f"CSV path:                 {output_csv_path}")
    print("=" * 60)

    return {
        "input_video": str(input_file),
        "output_tracking_video": str(output_video_path),
        "total_frames_processed": processed_frames,
        "total_unique_track_ids": len(unique_track_ids),
        "total_tracking_records": total_tracking_records,
        "average_processing_fps": avg_fps,
        "csv_path": str(output_csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 person tracking with ByteTrack on Avenue CCTV videos."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to Avenue video file (default: first discovered training video).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="YOLOv8 weights path or model name (default: yolov8x.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tracking",
        help="Directory to save tracking outputs (default: outputs/tracking).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="avenue_tracking.mp4",
        help="Output video filename (default: avenue_tracking.mp4).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="avenue_tracking.csv",
        help="Output CSV filename (default: avenue_tracking.csv).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration file (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: None for full video).",
    )

    args = parser.parse_args()

    track_video(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
        output_video_name=args.output_video,
        output_csv_name=args.output_csv,
        tracker_config=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
