import argparse
import csv
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from ultralytics import YOLO

from cluster_spatial import cluster_spatial_crowd
from track_clusters import track_persistent_clusters


def run_full_demo(
    video_path: str = "datasets/Avenue_Dataset/Avenue Dataset/training_videos/01.avi",
    model_path: str = "yolov8x.pt",
    output_dir: str = "outputs/clustering/full_video_demo",
    final_video_dest: str = "outputs/clustering/avenue_group_localization_full.mp4",
    conf: float = 0.15,
    imgsz: int = 640,
    eps: float = 60.0,
    min_samples: int = 3,
    small_group_eps: float = 60.0,
):
    print("=" * 80)
    print("FULL VIDEO CROWD CLUSTERING & GROUP LOCALIZATION DEMONSTRATION")
    print("=" * 80)

    in_file = Path(video_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input video not found: {in_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect source video
    cap = cv2.VideoCapture(str(in_file))
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    cap.release()

    print(f"Source Video:        {in_file}")
    print(f"Total Input Frames:  {total_input_frames}")
    print(f"FPS:                 {fps}")

    # 2. Run YOLO + ByteTrack on ALL frames
    tracking_csv_path = out_dir / "avenue_tracking_full.csv"
    print("\n--- STEP 1: Full-Video Detection & Tracking (YOLOv8 + ByteTrack) ---")
    t0 = time.time()

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(in_file))
    current_frame = 0
    tracking_rows = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            current_frame += 1

            if current_frame % 100 == 0 or current_frame == total_input_frames:
                elapsed = time.time() - t0
                speed_fps = current_frame / elapsed if elapsed > 0 else 0.0
                print(f"  Processed {current_frame}/{total_input_frames} frames ({current_frame/total_input_frames*100.0:.1f}%) | {speed_fps:.1f} FPS")

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                tids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for tid, conf_v, (x1, y1, x2, y2) in zip(tids, confs, xyxys):
                    bw = x2 - x1
                    bh = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    tracking_rows.append({
                        "frame": current_frame,
                        "track_id": tid,
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "width": f"{bw:.2f}",
                        "height": f"{bh:.2f}",
                        "center_x": f"{cx:.2f}",
                        "center_y": f"{cy:.2f}",
                        "foot_x": f"{foot_x:.2f}",
                        "foot_y": f"{foot_y:.2f}",
                        "confidence": f"{conf_v:.4f}",
                    })
    finally:
        cap.release()

    with open(tracking_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame", "track_id", "x1", "y1", "x2", "y2", "width", "height",
            "center_x", "center_y", "foot_x", "foot_y", "confidence",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tracking_rows)

    print(f"Tracking complete in {time.time()-t0:.1f}s. Total person detections: {len(tracking_rows)}")
    print(f"Saved: {tracking_csv_path}")

    # 3. Run Spatial DBSCAN Clustering with Small Group Disambiguation on ALL frames
    print("\n--- STEP 2: Spatial Clustering & Group Disambiguation (All Frames) ---")
    spatial_res = cluster_spatial_crowd(
        tracking_csv_path=tracking_csv_path,
        video_path=in_file,
        output_dir=out_dir,
        eps=eps,
        min_samples=min_samples,
        small_group_eps=small_group_eps,
        max_frames=None,
    )

    # 4. Run Persistent Cluster Tracking on ALL frames
    print("\n--- STEP 3: Persistent Group Tracking & Full-Body Localization Video ---")
    track_res = track_persistent_clusters(
        cluster_features_csv_path=out_dir / "cluster_features.csv",
        person_clusters_csv_path=out_dir / "person_clusters.csv",
        video_path=in_file,
        output_dir=out_dir,
        max_frames=None,
    )

    # 5. Copy final video to outputs/clustering/avenue_group_localization_full.mp4
    final_dest = Path(final_video_dest)
    final_dest.parent.mkdir(parents=True, exist_ok=True)
    generated_video = out_dir / "avenue_persistent_clusters.mp4"
    if generated_video.exists():
        shutil.copyfile(generated_video, final_dest)
        print(f"\nFinal Video Exported: {final_dest} ({final_dest.stat().st_size:,} bytes)")

    # Also make sure avenue_group_localization_full.mp4 is available in full_video_demo/
    demo_video = out_dir / "avenue_group_localization_full.mp4"
    if generated_video.exists():
        shutil.copyfile(generated_video, demo_video)

    # 6. Read back person_clusters.csv to compute exact summary statistics
    person_clusters_csv = out_dir / "person_clusters.csv"
    with open(person_clusters_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        p_rows = list(reader)

    total_detections = len(p_rows)
    crowd_detections = sum(1 for r in p_rows if r["group_type"] == "crowd_cluster")
    small_group_detections = sum(1 for r in p_rows if r["group_type"] == "small_group")
    isolated_detections = sum(1 for r in p_rows if r["group_type"] == "isolated_person")

    # Final summary output
    print("\n" + "=" * 80)
    print("FINAL FULL-VIDEO DEMONSTRATION RESULTS")
    print("=" * 80)
    print(f"Input frames:                {total_input_frames}")
    print(f"Processed frames:            {current_frame}")
    print(f"Total person detections:     {total_detections}")
    print(f"Crowd-cluster detections:    {crowd_detections}")
    print(f"Small-group detections:      {small_group_detections}")
    print(f"Isolated-person detections:  {isolated_detections}")
    print(f"Total crowd clusters:        {spatial_res['number_of_crowd_clusters']}")
    print(f"Total small groups:          {spatial_res['number_of_small_groups']}")
    print(f"Total isolated persons:      {spatial_res['number_of_isolated_persons']}")
    print()
    print(f"Output video:                {final_dest}")
    print(f"Output CSVs:                 {out_dir}")
    print("  - person/group membership: outputs/clustering/full_video_demo/person_clusters.csv")
    print("  - cluster/group features:  outputs/clustering/full_video_demo/cluster_features.csv")
    print("  - persistent groups:       outputs/clustering/full_video_demo/persistent_clusters.csv")
    print("  - persistent summary:      outputs/clustering/full_video_demo/persistent_cluster_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    run_full_demo()
