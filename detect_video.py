import argparse
from pathlib import Path
import time
import cv2
from ultralytics import YOLO


def discover_avenue_videos(
    base_dir: str | Path = "datasets/Avenue_Dataset/Avenue Dataset/training_videos",
) -> list[Path]:
    """Discover available Avenue video files (.avi) in the specified directory.

    Args:
        base_dir: Directory containing the Avenue video files.

    Returns:
        List of Path objects for discovered video files sorted alphabetically.
    """
    video_dir = Path(base_dir)
    if not video_dir.exists():
        return []
    return sorted(list(video_dir.glob("*.avi")))


def detect_video(
    video_path: str | Path | None = None,
    model_path: str | Path = "yolov8x.pt",
    output_dir: str | Path = "outputs/avenue_detection",
    conf: float = 0.15,
    imgsz: int = 640,
    max_frames: int | None = None,
) -> dict:
    """Run YOLOv8 person-only detection on a CCTV video frame-by-frame and save annotated video.

    Args:
        video_path: Path to the input video. If None, uses the first discovered training video.
        model_path: Path to the YOLOv8 model weights.
        output_dir: Directory where annotated output video will be saved.
        conf: Confidence threshold for person detection.
        imgsz: Inference image size.
        max_frames: Maximum number of frames to process (None for entire video).

    Returns:
        Dictionary containing execution metrics and summary details.
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
            # Check if provided as a relative filename inside default training folder
            fallback = Path("datasets/Avenue_Dataset/Avenue Dataset/training_videos") / input_file.name
            if fallback.exists():
                input_file = fallback
            else:
                raise FileNotFoundError(f"Input video file not found: {video_path}")

    # 2. Setup output paths
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = input_file.stem
    output_file = out_dir / f"{stem_name}_detected.mp4"

    # 3. Open input video stream
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_file}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN or zero check
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Initialize Windows-compatible VideoWriter (prefer MP4, fallback to AVI)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to XVID / AVI
        output_file = out_dir / f"{stem_name}_detected.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to initialize video writer for: {output_file}")

    # 5. Load YOLO model
    model = YOLO(str(model_path))

    # 6. Frame-by-frame processing
    processed_frames = 0
    total_person_detections = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # YOLO inference filtering only person class (COCO class 0)
            results = model(
                source=frame,
                classes=[0],
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )

            result = results[0]
            num_persons = len(result.boxes)
            total_person_detections += num_persons

            # Render bounding boxes and confidence scores
            annotated_frame = result.plot()

            # Write annotated frame
            writer.write(annotated_frame)
            processed_frames += 1

            if max_frames is not None and processed_frames >= max_frames:
                break

    finally:
        cap.release()
        writer.release()

    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

    # 7. Print summary metrics
    print("=" * 60)
    print("Avenue CCTV Video Detection Summary")
    print("=" * 60)
    print(f"Input video path:        {input_file}")
    print(f"Output video path:       {output_file}")
    print(f"Frame width:             {width}")
    print(f"Frame height:            {height}")
    print(f"Video FPS:               {fps:.2f}")
    print(f"Total video frames:      {total_frames}")
    print(f"Processed frames:        {processed_frames}")
    print(f"Total person detections: {total_person_detections}")
    print(f"Average processing FPS:  {avg_fps:.2f}")
    print("=" * 60)

    return {
        "input_video": str(input_file),
        "output_video": str(output_file),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_person_detections": total_person_detections,
        "average_fps": avg_fps,
        "output_file_exists": output_file.exists(),
        "output_file_size": output_file.stat().st_size if output_file.exists() else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 person detection on Avenue CCTV videos frame-by-frame."
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
        default="outputs/avenue_detection",
        help="Directory to save annotated video output (default: outputs/avenue_detection).",
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

    detect_video(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
