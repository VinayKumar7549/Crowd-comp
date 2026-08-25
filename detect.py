from pathlib import Path
import cv2
from ultralytics import YOLO


def detect_people(
    image_path: str | Path = "images/crowd.jpg",
    model_path: str | Path = "yolov8x.pt",
    output_path: str | Path = "outputs/detection_result.jpg",
    conf: float = 0.15,
    imgsz: int = 1280,
):
    """Detect people in an image using YOLOv8 (COCO class 0) and save the annotated result.

    Args:
        image_path: Path to the input image.
        model_path: Path or name of the YOLOv8 model weights.
        output_path: Path where the annotated image should be saved.
        conf: Detection confidence threshold.
        imgsz: Image inference size.

    Returns:
        tuple[int, Path]: Count of detected persons and the output file path.
    """
    input_file = Path(image_path)
    output_file = Path(output_path)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(str(model_path))

    # Run inference filtering only the person class (COCO class 0)
    results = model(
        source=str(input_file),
        classes=[0],
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    result = results[0]
    person_count = len(result.boxes)

    # Annotate image with bounding boxes
    annotated_img = result.plot()

    # Save annotated image
    cv2.imwrite(str(output_file), annotated_img)

    # Print required information
    print(f"Input image path: {input_file}")
    print(f"Number of people detected: {person_count}")
    print(f"Output image path: {output_file}")

    return person_count, output_file


if __name__ == "__main__":
    detect_people(
        image_path="images/crowd.jpg",
        model_path="yolov8x.pt",
        output_path="outputs/detection_result.jpg",
        conf=0.15,
        imgsz=1280,
    )