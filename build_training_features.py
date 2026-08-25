import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from detect_video import discover_avenue_videos


def wrap_angle_deg(angle_diff: float) -> float:
    """Wrap angular difference into range [-180, 180] degrees."""
    return (angle_diff + 180.0) % 360.0 - 180.0


# ==============================================================================
# SECTION N: OPTIONAL MODEL BENCHMARK
# ==============================================================================

def run_model_benchmark(
    video_path: str | Path,
    models: list[str] = ["yolov8n.pt", "yolov8x.pt"],
    num_frames: int = 100,
    conf: float = 0.15,
    imgsz: int = 640,
) -> list[dict]:
    """Benchmark YOLO model variants on representative CCTV frames."""
    print("=" * 70)
    print(f"YOLO Model Benchmark on {Path(video_path).name} ({num_frames} frames)")
    print("=" * 70)

    results = []
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Error: Could not read frames from video.")
        return results

    print(f"Loaded {len(frames)} frames into memory for clean benchmark.")
    print(f"{'Model':<14} | {'Processing FPS':<16} | {'Total Persons':<14} | {'Avg Conf':<10} | {'Time (s)':<10}")
    print("-" * 70)

    for m_name in models:
        try:
            model = YOLO(m_name)
            # Warm up
            model(frames[0], classes=[0], conf=conf, imgsz=imgsz, verbose=False)

            t0 = time.time()
            total_detections = 0
            confs_list = []

            for frame in frames:
                res = model(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    total_detections += len(res.boxes)
                    confs_list.extend(res.boxes.conf.cpu().tolist())

            elapsed = time.time() - t0
            proc_fps = len(frames) / elapsed if elapsed > 0 else 0.0
            avg_c = float(np.mean(confs_list)) if confs_list else 0.0

            res_dict = {
                "model": m_name,
                "processing_fps": proc_fps,
                "total_persons": total_detections,
                "avg_confidence": avg_c,
                "processing_time_s": elapsed,
            }
            results.append(res_dict)

            print(f"{m_name:<14} | {proc_fps:<16.2f} | {total_detections:<14} | {avg_c:<10.3f} | {elapsed:<10.2f}")
        except Exception as e:
            print(f"{m_name:<14} | Error loading/running model: {e}")

    print("-" * 70)
    return results


# ==============================================================================
# SECTION A & B: SINGLE VIDEO PROCESSING PIPELINE
# ==============================================================================

def process_single_video(
    video_path: Path,
    output_video_dir: Path,
    model: YOLO,
    conf: float = 0.15,
    imgsz: int = 640,
    dbscan_eps: float = 60.0,
    dbscan_min_samples: int = 3,
    start_frame: int = 1,
    end_frame: int | None = None,
    max_frames: int | None = None,
    save_video: bool = False,
    fps_override: float | None = None,
) -> dict:
    """Execute end-to-end detection -> tracking -> clustering -> feature extraction for one video."""
    video_stem = video_path.stem
    output_video_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    tracking_csv_path = output_video_dir / "person_tracking.csv"
    person_features_csv_path = output_video_dir / "person_motion_features.csv"
    person_summary_csv_path = output_video_dir / "person_motion_summary.csv"
    person_clusters_csv_path = output_video_dir / "person_clusters.csv"
    cluster_features_csv_path = output_video_dir / "cluster_features.csv"
    persistent_clusters_csv_path = output_video_dir / "persistent_clusters.csv"
    persistent_summary_csv_path = output_video_dir / "persistent_cluster_summary.csv"
    cluster_spatiotemporal_features_path = output_video_dir / "cluster_spatiotemporal_features.csv"
    cluster_spatiotemporal_summary_path = output_video_dir / "cluster_spatiotemporal_summary.csv"
    video_metadata_path = output_video_dir / "video_metadata.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps != src_fps:
        src_fps = 25.0
    effective_fps = fps_override if fps_override else src_fps
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optional video writer
    writer = None
    if save_video:
        annotated_video_path = output_video_dir / f"{video_stem}_spatiotemporal.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, effective_fps, (vid_w, vid_h))

    # --------------------------------------------------------------------------
    # Step 1: Frame-by-frame YOLOv8 Detection + ByteTrack
    # --------------------------------------------------------------------------
    tracking_rows = []
    raw_frame_people = defaultdict(list)
    unique_track_ids = set()

    current_frame = 0
    processed_frames = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            current_frame += 1

            if current_frame < start_frame:
                continue
            if end_frame is not None and current_frame > end_frame:
                break
            if max_frames is not None and processed_frames >= max_frames:
                break

            processed_frames += 1

            # Run YOLO + ByteTrack
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
                track_ids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for tid, conf_val, (x1, y1, x2, y2) in zip(track_ids, confs, xyxys):
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    obs = {
                        "frame": current_frame,
                        "track_id": tid,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": bbox_w,
                        "height": bbox_h,
                        "center_x": cx,
                        "center_y": cy,
                        "foot_x": foot_x,
                        "foot_y": foot_y,
                        "confidence": conf_val,
                    }
                    tracking_rows.append(obs)
                    raw_frame_people[current_frame].append(obs)
                    unique_track_ids.add(tid)

    finally:
        cap.release()

    # Save Person Tracking CSV
    with open(tracking_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame", "track_id", "x1", "y1", "x2", "y2", "width", "height",
            "center_x", "center_y", "foot_x", "foot_y", "confidence",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for r in tracking_rows:
            writer_csv.writerow({
                "frame": r["frame"],
                "track_id": r["track_id"],
                "x1": f"{r['x1']:.2f}",
                "y1": f"{r['y1']:.2f}",
                "x2": f"{r['x2']:.2f}",
                "y2": f"{r['y2']:.2f}",
                "width": f"{r['width']:.2f}",
                "height": f"{r['height']:.2f}",
                "center_x": f"{r['center_x']:.2f}",
                "center_y": f"{r['center_y']:.2f}",
                "foot_x": f"{r['foot_x']:.2f}",
                "foot_y": f"{r['foot_y']:.2f}",
                "confidence": f"{r['confidence']:.4f}",
            })

    # --------------------------------------------------------------------------
    # Step 2: Individual Motion Features Extraction
    # --------------------------------------------------------------------------
    tracks_by_id = defaultdict(list)
    for r in tracking_rows:
        tracks_by_id[r["track_id"]].append(r)

    person_motion_rows = []
    person_summary_rows = []
    person_motion_lookup = {}  # (frame, track_id) -> dict

    for tid, observations in sorted(tracks_by_id.items()):
        observations.sort(key=lambda x: x["frame"])
        if len(observations) < 2:
            continue

        prev_obs = None
        prev_vx = None
        prev_vy = None
        prev_dir = None
        speeds = []
        accels = []
        dir_changes = []
        step_disps = []

        for obs in observations:
            cf = obs["frame"]
            fx = obs["foot_x"]
            fy = obs["foot_y"]

            if prev_obs is None:
                dx = dy = disp = vx = vy = speed = accel = dir_change = 0.0
                dir_str = ""
            else:
                df = cf - prev_obs["frame"]
                dt = (df / effective_fps) if df > 0 else (1.0 / effective_fps)
                dx = fx - prev_obs["foot_x"]
                dy = fy - prev_obs["foot_y"]
                disp = math.sqrt(dx * dx + dy * dy)
                vx = dx / dt
                vy = dy / dt
                speed = disp / dt

                if prev_vx is not None and prev_vy is not None:
                    ax = (vx - prev_vx) / dt
                    ay = (vy - prev_vy) / dt
                    accel = math.sqrt(ax * ax + ay * ay)
                    accels.append(accel)
                else:
                    accel = 0.0

                if disp >= 0.5:
                    curr_d = math.degrees(math.atan2(dy, dx))
                    dir_str = f"{curr_d:.2f}"
                    if prev_dir is not None and prev_dir != "":
                        diff = abs(wrap_angle_deg(curr_d - float(prev_dir)))
                        dir_change = diff
                        dir_changes.append(diff)
                    else:
                        dir_change = 0.0
                    prev_dir = curr_d
                else:
                    dir_str = ""
                    dir_change = 0.0

                prev_vx = vx
                prev_vy = vy
                speeds.append(speed)
                step_disps.append(disp)

            prev_obs = obs

            pm_dict = {
                "frame": cf,
                "track_id": tid,
                "foot_x": fx,
                "foot_y": fy,
                "dx": dx,
                "dy": dy,
                "displacement": disp,
                "velocity_x": vx,
                "velocity_y": vy,
                "speed": speed,
                "acceleration": accel,
                "direction": dir_str,
                "direction_change": dir_change,
                "bbox_width": obs["width"],
                "bbox_height": obs["height"],
                "bbox_area": obs["width"] * obs["height"],
            }
            person_motion_rows.append(pm_dict)
            person_motion_lookup[(cf, tid)] = pm_dict

        # Track summary
        t_dur = (observations[-1]["frame"] - observations[0]["frame"]) / effective_fps
        net_d = math.sqrt(
            (observations[-1]["foot_x"] - observations[0]["foot_x"]) ** 2
            + (observations[-1]["foot_y"] - observations[0]["foot_y"]) ** 2
        )
        person_summary_rows.append({
            "track_id": tid,
            "track_duration": f"{t_dur:.2f}",
            "observation_count": len(observations),
            "trajectory_length": f"{sum(step_disps):.2f}",
            "total_displacement": f"{net_d:.2f}",
            "mean_speed": f"{np.mean(speeds):.2f}" if speeds else "0.00",
            "median_speed": f"{np.median(speeds):.2f}" if speeds else "0.00",
            "speed_std": f"{np.std(speeds):.2f}" if speeds else "0.00",
            "max_speed": f"{np.max(speeds):.2f}" if speeds else "0.00",
            "mean_acceleration": f"{np.mean(accels):.2f}" if accels else "0.00",
            "max_acceleration": f"{np.max(accels):.2f}" if accels else "0.00",
            "mean_direction_change": f"{np.mean(dir_changes):.2f}" if dir_changes else "0.00",
            "max_direction_change": f"{np.max(dir_changes):.2f}" if dir_changes else "0.00",
        })

    # Save Person Motion Features CSV
    with open(person_features_csv_path, mode="w", newline="", encoding="utf-8") as f:
        pm_fields = [
            "frame", "track_id", "foot_x", "foot_y", "dx", "dy", "displacement",
            "velocity_x", "velocity_y", "speed", "acceleration", "direction",
            "direction_change", "bbox_width", "bbox_height", "bbox_area",
        ]
        w = csv.DictWriter(f, fieldnames=pm_fields)
        w.writeheader()
        for r in person_motion_rows:
            w.writerow({
                "frame": r["frame"],
                "track_id": r["track_id"],
                "foot_x": f"{r['foot_x']:.2f}",
                "foot_y": f"{r['foot_y']:.2f}",
                "dx": f"{r['dx']:.2f}",
                "dy": f"{r['dy']:.2f}",
                "displacement": f"{r['displacement']:.2f}",
                "velocity_x": f"{r['velocity_x']:.2f}",
                "velocity_y": f"{r['velocity_y']:.2f}",
                "speed": f"{r['speed']:.2f}",
                "acceleration": f"{r['acceleration']:.2f}",
                "direction": r["direction"],
                "direction_change": f"{r['direction_change']:.2f}",
                "bbox_width": f"{r['bbox_width']:.2f}",
                "bbox_height": f"{r['bbox_height']:.2f}",
                "bbox_area": f"{r['bbox_area']:.2f}",
            })

    # --------------------------------------------------------------------------
    # Step 3: DBSCAN Spatial Clustering & Small Group Disambiguation
    # --------------------------------------------------------------------------
    person_cluster_rows = []
    frame_clusters_map = defaultdict(list)
    cluster_member_tracks = defaultdict(list)

    for cf in sorted(raw_frame_people.keys()):
        people = raw_frame_people[cf]
        if len(people) < dbscan_min_samples:
            labels = np.full(len(people), -1, dtype=int)
        else:
            coords = np.array([[p["foot_x"], p["foot_y"]] for p in people])
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords)
            labels = db.labels_

        # Process Crowd Clusters (>= dbscan_min_samples)
        unique_cids = sorted([cid for cid in set(labels) if cid != -1])
        for cid in unique_cids:
            members = [p for p, l in zip(people, labels) if l == cid]
            n_p = len(members)
            xs = [m["foot_x"] for m in members]
            ys = [m["foot_y"] for m in members]

            # Full body bounding box covering all member visible bodies
            c_x1 = float(min(m["x1"] for m in members))
            c_y1 = float(min(m["y1"] for m in members))
            c_x2 = float(max(m["x2"] for m in members))
            c_y2 = float(max(m["y2"] for m in members))

            w = c_x2 - c_x1
            h = c_y2 - c_y1
            area = w * h
            density = (n_p / area) if area > 0 else 0.0

            c_dict = {
                "frame": cf,
                "group_type": "crowd_cluster",
                "group_id": cid,
                "dbscan_cluster_id": cid,
                "number_of_people": n_p,
                "centroid_x": float(np.mean(xs)),
                "centroid_y": float(np.mean(ys)),
                "min_x": float(np.min(xs)),
                "max_x": float(np.max(xs)),
                "min_y": float(np.min(ys)),
                "max_y": float(np.max(ys)),
                "cluster_x1": c_x1,
                "cluster_y1": c_y1,
                "cluster_x2": c_x2,
                "cluster_y2": c_y2,
                "width": w,
                "height": h,
                "area": area,
                "density": density,
                "members": set([m["track_id"] for m in members]),
            }
            frame_clusters_map[cf].append(c_dict)

            for m in members:
                m["group_type"] = "crowd_cluster"
                m["group_id"] = cid
                m["group_size"] = n_p
                m["cluster_x1"] = c_x1
                m["cluster_y1"] = c_y1
                m["cluster_x2"] = c_x2
                m["cluster_y2"] = c_y2
                m["cluster_id"] = cid
                cluster_member_tracks[(cf, int(cid))].append(m["track_id"])

        # Process Noise Points -> Small Groups or Isolated Persons
        noise_indices = [i for i, l in enumerate(labels) if l == -1]
        if len(noise_indices) >= 2:
            noise_people = [people[i] for i in noise_indices]
            noise_coords = np.array([[p["foot_x"], p["foot_y"]] for p in noise_people])
            sg_db = DBSCAN(eps=dbscan_eps, min_samples=2).fit(noise_coords)
            sg_labels = sg_db.labels_

            unique_sg_ids = sorted([sg for sg in set(sg_labels) if sg != -1])
            for sg_id in unique_sg_ids:
                sg_members = [p for p, l in zip(noise_people, sg_labels) if l == sg_id]
                n_sg = len(sg_members)
                xs = [m["foot_x"] for m in sg_members]
                ys = [m["foot_y"] for m in sg_members]

                sg_x1 = float(min(m["x1"] for m in sg_members))
                sg_y1 = float(min(m["y1"] for m in sg_members))
                sg_x2 = float(max(m["x2"] for m in sg_members))
                sg_y2 = float(max(m["y2"] for m in sg_members))

                w = sg_x2 - sg_x1
                h = sg_y2 - sg_y1
                area = w * h
                density = (n_sg / area) if area > 0 else 0.0

                sg_dict = {
                    "frame": cf,
                    "group_type": "small_group",
                    "group_id": f"SG_{sg_id}",
                    "dbscan_cluster_id": -1,
                    "number_of_people": n_sg,
                    "centroid_x": float(np.mean(xs)),
                    "centroid_y": float(np.mean(ys)),
                    "min_x": float(np.min(xs)),
                    "max_x": float(np.max(xs)),
                    "min_y": float(np.min(ys)),
                    "max_y": float(np.max(ys)),
                    "cluster_x1": sg_x1,
                    "cluster_y1": sg_y1,
                    "cluster_x2": sg_x2,
                    "cluster_y2": sg_y2,
                    "width": w,
                    "height": h,
                    "area": area,
                    "density": density,
                    "members": set([m["track_id"] for m in sg_members]),
                }
                frame_clusters_map[cf].append(sg_dict)

                for m in sg_members:
                    m["group_type"] = "small_group"
                    m["group_id"] = f"SG_{sg_id}"
                    m["group_size"] = n_sg
                    m["cluster_x1"] = sg_x1
                    m["cluster_y1"] = sg_y1
                    m["cluster_x2"] = sg_x2
                    m["cluster_y2"] = sg_y2
                    m["cluster_id"] = -1

            for p, l in zip(noise_people, sg_labels):
                if l == -1:
                    p["group_type"] = "isolated_person"
                    p["group_id"] = p["track_id"]
                    p["group_size"] = 1
                    p["cluster_x1"] = p["x1"]
                    p["cluster_y1"] = p["y1"]
                    p["cluster_x2"] = p["x2"]
                    p["cluster_y2"] = p["y2"]
                    p["cluster_id"] = -1
        elif len(noise_indices) == 1:
            p = people[noise_indices[0]]
            p["group_type"] = "isolated_person"
            p["group_id"] = p["track_id"]
            p["group_size"] = 1
            p["cluster_x1"] = p["x1"]
            p["cluster_y1"] = p["y1"]
            p["cluster_x2"] = p["x2"]
            p["cluster_y2"] = p["y2"]
            p["cluster_id"] = -1

        for person in people:
            person_cluster_rows.append({
                "frame": cf,
                "track_id": person["track_id"],
                "foot_x": person["foot_x"],
                "foot_y": person["foot_y"],
                "cluster_id": person.get("cluster_id", -1),
                "group_type": person["group_type"],
                "group_id": person["group_id"],
                "group_size": person["group_size"],
                "cluster_x1": person["cluster_x1"],
                "cluster_y1": person["cluster_y1"],
                "cluster_x2": person["cluster_x2"],
                "cluster_y2": person["cluster_y2"],
            })

    # Save Person Clusters CSV
    with open(person_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
        person_fields = [
            "frame", "track_id", "foot_x", "foot_y", "cluster_id",
            "group_type", "group_id", "group_size",
            "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
        ]
        w = csv.DictWriter(f, fieldnames=person_fields)
        w.writeheader()
        for r in person_cluster_rows:
            w.writerow({
                "frame": r["frame"],
                "track_id": r["track_id"],
                "foot_x": f"{r['foot_x']:.2f}",
                "foot_y": f"{r['foot_y']:.2f}",
                "cluster_id": r["cluster_id"],
                "group_type": r["group_type"],
                "group_id": r["group_id"],
                "group_size": r["group_size"],
                "cluster_x1": f"{r['cluster_x1']:.2f}",
                "cluster_y1": f"{r['cluster_y1']:.2f}",
                "cluster_x2": f"{r['cluster_x2']:.2f}",
                "cluster_y2": f"{r['cluster_y2']:.2f}",
            })

    # --------------------------------------------------------------------------
    # Step 4: Persistent Cluster Association
    # --------------------------------------------------------------------------
    active_cluster_tracks = {}
    next_pid = 0
    persistent_cluster_records = []
    pid_history = defaultdict(list)

    for cf in sorted(raw_frame_people.keys()):
        curr_c_list = frame_clusters_map.get(cf, [])

        if not active_cluster_tracks and curr_c_list:
            for c in curr_c_list:
                pid = next_pid
                next_pid += 1
                active_cluster_tracks[pid] = {
                    "persistent_id": pid,
                    "last_frame": cf,
                    "centroid_x": c["centroid_x"],
                    "centroid_y": c["centroid_y"],
                    "members": c["members"],
                    "missed_frames": 0,
                }
                c["persistent_cluster_id"] = pid
                persistent_cluster_records.append(c)
                pid_history[pid].append(c)

        elif not curr_c_list:
            for pid, trk in list(active_cluster_tracks.items()):
                trk["missed_frames"] += 1
                if trk["missed_frames"] > 5:
                    del active_cluster_tracks[pid]

        else:
            active_pids = list(active_cluster_tracks.keys())
            cost_mat = np.full((len(active_pids), len(curr_c_list)), 1e6, dtype=float)

            for i, pid in enumerate(active_pids):
                trk = active_cluster_tracks[pid]
                for j, c in enumerate(curr_c_list):
                    dist = math.sqrt((c["centroid_x"] - trk["centroid_x"]) ** 2 + (c["centroid_y"] - trk["centroid_y"]) ** 2)
                    union_m = trk["members"] | c["members"]
                    overlap = (len(trk["members"] & c["members"]) / len(union_m)) if union_m else 0.0
                    cost = dist * (1.0 - 0.5 * overlap)

                    if dist <= 80.0 or (overlap >= 0.20 and dist <= 120.0):
                        cost_mat[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_act = set()
            matched_cur = set()

            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e5:
                    pid = active_pids[r]
                    curr_c = curr_c_list[c]
                    curr_c["persistent_cluster_id"] = pid
                    active_cluster_tracks[pid]["last_frame"] = cf
                    active_cluster_tracks[pid]["centroid_x"] = curr_c["centroid_x"]
                    active_cluster_tracks[pid]["centroid_y"] = curr_c["centroid_y"]
                    active_cluster_tracks[pid]["members"] = curr_c["members"]
                    active_cluster_tracks[pid]["missed_frames"] = 0

                    matched_act.add(pid)
                    matched_cur.add(c)
                    persistent_cluster_records.append(curr_c)
                    pid_history[pid].append(curr_c)

            for pid in active_pids:
                if pid not in matched_act:
                    active_cluster_tracks[pid]["missed_frames"] += 1
                    if active_cluster_tracks[pid]["missed_frames"] > 5:
                        del active_cluster_tracks[pid]

            for j, c in enumerate(curr_c_list):
                if j not in matched_cur:
                    pid = next_pid
                    next_pid += 1
                    active_cluster_tracks[pid] = {
                        "persistent_id": pid,
                        "last_frame": cf,
                        "centroid_x": c["centroid_x"],
                        "centroid_y": c["centroid_y"],
                        "members": c["members"],
                        "missed_frames": 0,
                    }
                    c["persistent_cluster_id"] = pid
                    persistent_cluster_records.append(c)
                    pid_history[pid].append(c)

    # Save Persistent Clusters CSV
    with open(persistent_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
        pers_cols = [
            "frame", "group_type", "group_id", "persistent_cluster_id", "dbscan_cluster_id", "number_of_people",
            "centroid_x", "centroid_y", "min_x", "max_x", "min_y", "max_y",
            "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
            "width", "height", "area", "density",
        ]
        w = csv.DictWriter(f, fieldnames=pers_cols, extrasaction="ignore")
        w.writeheader()
        for r in persistent_cluster_records:
            w.writerow({
                "frame": r["frame"],
                "group_type": r.get("group_type", "crowd_cluster"),
                "group_id": r.get("group_id", r.get("dbscan_cluster_id")),
                "persistent_cluster_id": r["persistent_cluster_id"],
                "dbscan_cluster_id": r["dbscan_cluster_id"],
                "number_of_people": r["number_of_people"],
                "centroid_x": f"{r['centroid_x']:.2f}",
                "centroid_y": f"{r['centroid_y']:.2f}",
                "min_x": f"{r['min_x']:.2f}",
                "max_x": f"{r['max_x']:.2f}",
                "min_y": f"{r['min_y']:.2f}",
                "max_y": f"{r['max_y']:.2f}",
                "cluster_x1": f"{r.get('cluster_x1', r['min_x']):.2f}",
                "cluster_y1": f"{r.get('cluster_y1', r['min_y']):.2f}",
                "cluster_x2": f"{r.get('cluster_x2', r['max_x']):.2f}",
                "cluster_y2": f"{r.get('cluster_y2', r['max_y']):.2f}",
                "width": f"{r['width']:.2f}",
                "height": f"{r['height']:.2f}",
                "area": f"{r['area']:.2f}",
                "density": f"{r['density']:.6f}",
            })

    # --------------------------------------------------------------------------
    # Step 5: Cluster-Level Spatiotemporal Discriminant Features
    # --------------------------------------------------------------------------
    cluster_spatiotemporal_rows = []
    cluster_summary_rows = []

    for pid, observations in sorted(pid_history.items()):
        observations.sort(key=lambda x: x["frame"])

        prev_obs = None
        prev_vx = None
        prev_vy = None
        prev_dir = None

        speeds = []
        accels = []
        dir_changes = []
        densities = []
        coherences = []
        areas = []
        dispersions = []
        people_counts = []
        displacements = []
        centroids = []

        for obs in observations:
            cf = obs["frame"]
            dbscan_id = obs["dbscan_cluster_id"]
            n_p = obs["number_of_people"]
            cx = obs["centroid_x"]
            cy = obs["centroid_y"]
            cw = obs["width"]
            ch = obs["height"]
            c_area = obs["area"]
            c_dens = obs["density"]

            centroids.append((cx, cy))
            people_counts.append(n_p)
            areas.append(c_area)
            densities.append(c_dens)

            if prev_obs is None:
                cdx = cdy = c_disp = c_vx = c_vy = c_speed = c_accel = 0.0
                dir_str = ""
                dir_change = 0.0
                area_change = width_change = height_change = density_change = 0.0
            else:
                df = cf - prev_obs["frame"]
                dt = (df / effective_fps) if df > 0 else (1.0 / effective_fps)
                cdx = cx - prev_obs["centroid_x"]
                cdy = cy - prev_obs["centroid_y"]
                c_disp = math.sqrt(cdx * cdx + cdy * cdy)
                c_vx = cdx / dt
                c_vy = cdy / dt
                c_speed = c_disp / dt

                if prev_vx is not None and prev_vy is not None:
                    ax = (c_vx - prev_vx) / dt
                    ay = (c_vy - prev_vy) / dt
                    c_accel = math.sqrt(ax * ax + ay * ay)
                    accels.append(c_accel)
                else:
                    c_accel = 0.0

                if c_disp >= 0.5:
                    curr_d = math.degrees(math.atan2(cdy, cdx))
                    dir_str = f"{curr_d:.2f}"
                    if prev_dir is not None and prev_dir != "":
                        dir_change = abs(wrap_angle_deg(curr_d - float(prev_dir)))
                        dir_changes.append(dir_change)
                    else:
                        dir_change = 0.0
                    prev_dir = curr_d
                else:
                    dir_str = ""
                    dir_change = 0.0

                prev_vx = c_vx
                prev_vy = c_vy
                area_change = (c_area - prev_obs["area"]) / dt
                width_change = (cw - prev_obs["width"]) / dt
                height_change = (ch - prev_obs["height"]) / dt
                density_change = (c_dens - prev_obs["density"]) / dt

                speeds.append(c_speed)
                displacements.append(c_disp)

            prev_obs = obs

            # Member pedestrian dispersion & coherence
            member_tids = cluster_member_tracks.get((cf, dbscan_id), [])
            dists = []
            m_speeds = []
            m_vxs = []
            m_vys = []
            uvx_list = []
            uvy_list = []

            for tid in member_tids:
                p_mot = person_motion_lookup.get((cf, tid))
                if p_mot:
                    px, py = p_mot["foot_x"], p_mot["foot_y"]
                    d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                    dists.append(d)

                    vx_i = p_mot["velocity_x"]
                    vy_i = p_mot["velocity_y"]
                    s_i = p_mot["speed"]
                    m_speeds.append(s_i)
                    m_vxs.append(vx_i)
                    m_vys.append(vy_i)

                    if s_i >= 1.0:
                        uvx_list.append(vx_i / s_i)
                        uvy_list.append(vy_i / s_i)

            if dists:
                mean_dist = float(np.mean(dists))
                med_dist = float(np.median(dists))
                std_dist = float(np.std(dists))
                max_dist = float(np.max(dists))
                dispersion = float(np.sqrt(np.mean(np.square(dists))))
            else:
                mean_dist = med_dist = std_dist = max_dist = dispersion = 0.0
            dispersions.append(dispersion)

            if m_speeds:
                mean_s = float(np.mean(m_speeds))
                std_s = float(np.std(m_speeds))
                max_s = float(np.max(m_speeds))
                m_vx = float(np.mean(m_vxs))
                m_vy = float(np.mean(m_vys))
                v_disp = float(np.sqrt(np.mean([(vx_i - m_vx) ** 2 + (vy_i - m_vy) ** 2 for vx_i, vy_i in zip(m_vxs, m_vys)])))
            else:
                mean_s = std_s = max_s = m_vx = m_vy = v_disp = 0.0

            if uvx_list:
                mu_x, mu_y = float(np.mean(uvx_list)), float(np.mean(uvy_list))
                coherence = float(np.clip(math.sqrt(mu_x * mu_x + mu_y * mu_y), 0.0, 1.0))
            else:
                coherence = 1.0
            coherences.append(coherence)

            row = {
                "frame": cf,
                "persistent_cluster_id": pid,
                "number_of_people": n_p,
                "centroid_x": f"{cx:.2f}",
                "centroid_y": f"{cy:.2f}",
                "centroid_dx": f"{cdx:.2f}",
                "centroid_dy": f"{cdy:.2f}",
                "cluster_displacement": f"{c_disp:.2f}",
                "cluster_velocity_x": f"{c_vx:.2f}",
                "cluster_velocity_y": f"{c_vy:.2f}",
                "cluster_speed": f"{c_speed:.2f}",
                "cluster_acceleration": f"{c_accel:.2f}",
                "cluster_direction": dir_str,
                "cluster_direction_change": f"{dir_change:.2f}",
                "cluster_width": f"{cw:.2f}",
                "cluster_height": f"{ch:.2f}",
                "cluster_area": f"{c_area:.2f}",
                "density": f"{c_dens:.6f}",
                "mean_distance_to_centroid": f"{mean_dist:.2f}",
                "median_distance_to_centroid": f"{med_dist:.2f}",
                "std_distance_to_centroid": f"{std_dist:.2f}",
                "max_distance_to_centroid": f"{max_dist:.2f}",
                "spatial_dispersion": f"{dispersion:.2f}",
                "mean_individual_speed": f"{mean_s:.2f}",
                "std_individual_speed": f"{std_s:.2f}",
                "max_individual_speed": f"{max_s:.2f}",
                "mean_velocity_x": f"{m_vx:.2f}",
                "mean_velocity_y": f"{m_vy:.2f}",
                "velocity_dispersion": f"{v_disp:.2f}",
                "directional_coherence": f"{coherence:.4f}",
                "area_change": f"{area_change:.2f}",
                "width_change": f"{width_change:.2f}",
                "height_change": f"{height_change:.2f}",
                "density_change": f"{density_change:.6f}",
            }
            cluster_spatiotemporal_rows.append(row)

        # Lifetime summary
        first_f = observations[0]["frame"]
        last_f = observations[-1]["frame"]
        dur_f = last_f - first_f + 1
        dur_s = dur_f / effective_fps
        t_len = sum(displacements)
        net_d = math.sqrt((centroids[-1][0] - centroids[0][0]) ** 2 + (centroids[-1][1] - centroids[0][1]) ** 2)
        t_eff = (net_d / t_len) if t_len > 0 else 0.0

        cluster_summary_rows.append({
            "persistent_cluster_id": pid,
            "first_frame": first_f,
            "last_frame": last_f,
            "duration_frames": dur_f,
            "duration_seconds": f"{dur_s:.2f}",
            "mean_people": f"{np.mean(people_counts):.2f}",
            "max_people": int(np.max(people_counts)),
            "min_people": int(np.min(people_counts)),
            "mean_speed": f"{np.mean(speeds):.2f}" if speeds else "0.00",
            "median_speed": f"{np.median(speeds):.2f}" if speeds else "0.00",
            "speed_std": f"{np.std(speeds):.2f}" if speeds else "0.00",
            "max_speed": f"{np.max(speeds):.2f}" if speeds else "0.00",
            "mean_acceleration": f"{np.mean(accels):.2f}" if accels else "0.00",
            "max_acceleration": f"{np.max(accels):.2f}" if accels else "0.00",
            "mean_direction_change": f"{np.mean(dir_changes):.2f}" if dir_changes else "0.00",
            "max_direction_change": f"{np.max(dir_changes):.2f}" if dir_changes else "0.00",
            "mean_density": f"{np.mean(densities):.6f}",
            "density_std": f"{np.std(densities):.6f}",
            "mean_directional_coherence": f"{np.mean(coherences):.4f}",
            "min_directional_coherence": f"{np.min(coherences):.4f}",
            "max_directional_coherence": f"{np.max(coherences):.4f}",
            "mean_cluster_area": f"{np.mean(areas):.2f}",
            "max_cluster_area": f"{np.max(areas):.2f}",
            "min_cluster_area": f"{np.min(areas):.2f}",
            "mean_dispersion": f"{np.mean(dispersions):.2f}",
            "max_dispersion": f"{np.max(dispersions):.2f}",
            "total_centroid_trajectory_length": f"{t_len:.2f}",
            "net_centroid_displacement": f"{net_d:.2f}",
            "trajectory_efficiency": f"{t_eff:.4f}",
        })

    # Save Cluster Spatiotemporal Features CSV
    with open(cluster_spatiotemporal_features_path, mode="w", newline="", encoding="utf-8") as f:
        c_fields = [
            "frame", "persistent_cluster_id", "number_of_people", "centroid_x", "centroid_y",
            "centroid_dx", "centroid_dy", "cluster_displacement", "cluster_velocity_x", "cluster_velocity_y",
            "cluster_speed", "cluster_acceleration", "cluster_direction", "cluster_direction_change",
            "cluster_width", "cluster_height", "cluster_area", "density",
            "mean_distance_to_centroid", "median_distance_to_centroid", "std_distance_to_centroid", "max_distance_to_centroid",
            "spatial_dispersion", "mean_individual_speed", "std_individual_speed", "max_individual_speed",
            "mean_velocity_x", "mean_velocity_y", "velocity_dispersion", "directional_coherence",
            "area_change", "width_change", "height_change", "density_change",
        ]
        w = csv.DictWriter(f, fieldnames=c_fields)
        w.writeheader()
        w.writerows(cluster_spatiotemporal_rows)

    # Save Cluster Summary CSV
    with open(cluster_spatiotemporal_summary_path, mode="w", newline="", encoding="utf-8") as f:
        s_fields = [
            "persistent_cluster_id", "first_frame", "last_frame", "duration_frames", "duration_seconds",
            "mean_people", "max_people", "min_people", "mean_speed", "median_speed", "speed_std", "max_speed",
            "mean_acceleration", "max_acceleration", "mean_direction_change", "max_direction_change",
            "mean_density", "density_std", "mean_directional_coherence", "min_directional_coherence", "max_directional_coherence",
            "mean_cluster_area", "max_cluster_area", "min_cluster_area", "mean_dispersion", "max_dispersion",
            "total_centroid_trajectory_length", "net_centroid_displacement", "trajectory_efficiency",
        ]
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(cluster_summary_rows)

    elapsed_time = time.time() - t_start
    proc_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

    meta = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "source_resolution": f"{vid_w}x{vid_h}",
        "source_fps": effective_fps,
        "total_source_frames": total_vid_frames,
        "processed_frames": processed_frames,
        "processing_time_s": elapsed_time,
        "processing_fps": proc_fps,
        "total_person_detections": len(tracking_rows),
        "unique_person_tracks": len(unique_track_ids),
        "persistent_clusters_count": len(cluster_summary_rows),
        "cluster_feature_rows_count": len(cluster_spatiotemporal_rows),
    }

    with open(video_metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


# ==============================================================================
# SECTION F, G, H, I, J, K: MERGE & ANALYSIS ACROSS ALL VIDEOS
# ==============================================================================

def merge_and_analyze_dataset(
    output_base_dir: Path,
    config_dict: dict,
) -> dict:
    """Merge all video features into unified normal dataset CSVs, generate plots, correlation, and metadata."""
    print("\n" + "=" * 70)
    print("Merging All Training Video Features into Normal Training Baseline")
    print("=" * 70)

    merged_features_csv = output_base_dir / "normal_cluster_features.csv"
    merged_summary_csv = output_base_dir / "normal_cluster_summary.csv"
    distributions_dir = output_base_dir / "normal_distributions"
    distributions_dir.mkdir(parents=True, exist_ok=True)
    corr_img_path = output_base_dir / "normal_feature_correlation.png"
    corr_csv_path = output_base_dir / "normal_feature_correlation.csv"
    dataset_metadata_path = output_base_dir / "normal_dataset_metadata.json"

    # 1. Discover all processed video directories
    video_dirs = sorted([d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith("video_")])
    if not video_dirs:
        raise FileNotFoundError("No processed video directories found in output directory.")

    all_merged_feature_rows = []
    all_merged_summary_rows = []

    total_detections_all = 0
    unique_tracks_all = 0
    processed_frames_all = 0

    for v_dir in video_dirs:
        v_id = v_dir.name  # e.g. 'video_01'
        feat_path = v_dir / "cluster_spatiotemporal_features.csv"
        sum_path = v_dir / "cluster_spatiotemporal_summary.csv"
        meta_path = v_dir / "video_metadata.json"

        if meta_path.exists():
            with open(meta_path, mode="r", encoding="utf-8") as f:
                v_meta = json.load(f)
                total_detections_all += v_meta.get("total_person_detections", 0)
                unique_tracks_all += v_meta.get("unique_person_tracks", 0)
                processed_frames_all += v_meta.get("processed_frames", 0)

        if feat_path.exists():
            with open(feat_path, mode="r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    all_merged_feature_rows.append({"video_id": v_id, **row})

        if sum_path.exists():
            with open(sum_path, mode="r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    all_merged_summary_rows.append({"video_id": v_id, **row})

    # Write merged normal_cluster_features.csv
    if all_merged_feature_rows:
        feat_cols = ["video_id"] + [k for k in all_merged_feature_rows[0].keys() if k != "video_id"]
        with open(merged_features_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=feat_cols)
            w.writeheader()
            w.writerows(all_merged_feature_rows)

    # Write merged normal_cluster_summary.csv
    if all_merged_summary_rows:
        sum_cols = ["video_id"] + [k for k in all_merged_summary_rows[0].keys() if k != "video_id"]
        with open(merged_summary_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sum_cols)
            w.writeheader()
            w.writerows(all_merged_summary_rows)

    # --------------------------------------------------------------------------
    # Step H: Feature Quality Analysis
    # --------------------------------------------------------------------------
    nan_count = 0
    inf_count = 0
    missing_count = 0
    duplicate_count = 0

    seen_keys = set()
    for row in all_merged_feature_rows:
        key = (row["video_id"], row["frame"], row["persistent_cluster_id"])
        if key in seen_keys:
            duplicate_count += 1
        seen_keys.add(key)

        for k, v in row.items():
            if v is None or v == "":
                if k not in ["cluster_direction"]:  # direction can be blank when stationary
                    missing_count += 1
            else:
                try:
                    val = float(v)
                    if math.isnan(val):
                        nan_count += 1
                    if math.isinf(val):
                        inf_count += 1
                except ValueError:
                    pass

    # --------------------------------------------------------------------------
    # Step I: Normal Feature Distributions & Percentiles (10 candidate features)
    # --------------------------------------------------------------------------
    candidate_features = [
        ("cluster_speed", "Cluster Speed (px/s)", "cluster_speed"),
        ("cluster_acceleration", "Cluster Acceleration (px/s²)", "cluster_acceleration"),
        ("density", "Cluster Density (people/px²)", "density"),
        ("directional_coherence", "Directional Coherence [0, 1]", "directional_coherence"),
        ("spatial_dispersion", "Spatial Dispersion (px)", "spatial_dispersion"),
        ("area_change", "Area Expansion/Contraction (px²/s)", "area_change"),
        ("density_change", "Density Change Rate", "density_change"),
        ("cluster_direction_change", "Direction Change (°)", "direction_change"),
        ("number_of_people", "People per Cluster", "number_of_people"),
    ]

    # Add trajectory_efficiency from summary table
    traj_effs = [float(r["trajectory_efficiency"]) for r in all_merged_summary_rows]

    percentiles_table = []

    for feat_key, feat_label, fname in candidate_features:
        values = [float(r[feat_key]) for r in all_merged_feature_rows if r.get(feat_key) not in ["", None]]
        if values:
            p1 = np.percentile(values, 1)
            p5 = np.percentile(values, 5)
            p25 = np.percentile(values, 25)
            p50 = np.percentile(values, 50)
            p75 = np.percentile(values, 75)
            p95 = np.percentile(values, 95)
            p99 = np.percentile(values, 99)
            mean_v = np.mean(values)
            std_v = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)

            percentiles_table.append({
                "feature": feat_key,
                "mean": f"{mean_v:.4f}",
                "std": f"{std_v:.4f}",
                "min": f"{min_v:.4f}",
                "p1": f"{p1:.4f}",
                "p5": f"{p5:.4f}",
                "p25": f"{p25:.4f}",
                "median_p50": f"{p50:.4f}",
                "p75": f"{p75:.4f}",
                "p95": f"{p95:.4f}",
                "p99": f"{p99:.4f}",
                "max": f"{max_v:.4f}",
            })

            # Plot distribution
            plt.figure(figsize=(7.5, 4.5))
            plt.hist(values, bins=30, color="#2563EB", edgecolor="black", alpha=0.75)
            plt.axvline(mean_v, color="#DC2626", linestyle="--", label=f"Mean: {mean_v:.2f}")
            plt.axvline(p50, color="#16A34A", linestyle=":", label=f"Median: {p50:.2f}")
            plt.title(f"Normal Training Distribution: {feat_label}", fontsize=12, fontweight="bold")
            plt.xlabel(feat_label)
            plt.ylabel("Observation Count")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(distributions_dir / f"{fname}_dist.png", dpi=150)
            plt.close()

    # Trajectory efficiency distribution
    if traj_effs:
        p1 = np.percentile(traj_effs, 1)
        p5 = np.percentile(traj_effs, 5)
        p25 = np.percentile(traj_effs, 25)
        p50 = np.percentile(traj_effs, 50)
        p75 = np.percentile(traj_effs, 75)
        p95 = np.percentile(traj_effs, 95)
        p99 = np.percentile(traj_effs, 99)
        mean_v = np.mean(traj_effs)
        std_v = np.std(traj_effs)
        min_v = np.min(traj_effs)
        max_v = np.max(traj_effs)

        percentiles_table.append({
            "feature": "trajectory_efficiency",
            "mean": f"{mean_v:.4f}",
            "std": f"{std_v:.4f}",
            "min": f"{min_v:.4f}",
            "p1": f"{p1:.4f}",
            "p5": f"{p5:.4f}",
            "p25": f"{p25:.4f}",
            "median_p50": f"{p50:.4f}",
            "p75": f"{p75:.4f}",
            "p95": f"{p95:.4f}",
            "p99": f"{p99:.4f}",
            "max": f"{max_v:.4f}",
        })

        plt.figure(figsize=(7.5, 4.5))
        plt.hist(traj_effs, bins=20, color="#7C3AED", edgecolor="black", alpha=0.75)
        plt.axvline(mean_v, color="#DC2626", linestyle="--", label=f"Mean: {mean_v:.2f}")
        plt.axvline(p50, color="#16A34A", linestyle=":", label=f"Median: {p50:.2f}")
        plt.title("Normal Training Distribution: Trajectory Efficiency", fontsize=12, fontweight="bold")
        plt.xlabel("Trajectory Efficiency (Net Disp / Trajectory Length)")
        plt.ylabel("Cluster Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(distributions_dir / "trajectory_efficiency_dist.png", dpi=150)
        plt.close()

    # Save Percentiles CSV
    with open(distributions_dir / "normal_feature_percentiles.csv", mode="w", newline="", encoding="utf-8") as f:
        pct_fields = ["feature", "mean", "std", "min", "p1", "p5", "p25", "median_p50", "p75", "p95", "p99", "max"]
        w = csv.DictWriter(f, fieldnames=pct_fields)
        w.writeheader()
        w.writerows(percentiles_table)

    # --------------------------------------------------------------------------
    # Step J: Feature Correlation Matrix
    # --------------------------------------------------------------------------
    corr_keys = [
        ("number_of_people", "People Count"),
        ("cluster_speed", "Speed"),
        ("cluster_acceleration", "Acceleration"),
        ("cluster_area", "Area"),
        ("density", "Density"),
        ("spatial_dispersion", "Dispersion"),
        ("directional_coherence", "Coherence"),
        ("velocity_dispersion", "Vel Dispersion"),
    ]
    feat_matrix = []
    for row in all_merged_feature_rows:
        feat_matrix.append([float(row[k]) for k, _ in corr_keys])

    if feat_matrix:
        arr = np.array(feat_matrix, dtype=float)
        corr_mat = np.corrcoef(arr, rowvar=False)

        # Save Correlation CSV
        with open(corr_csv_path, mode="w", newline="", encoding="utf-8") as f:
            c_writer = csv.writer(f)
            c_writer.writerow(["feature"] + [lbl for _, lbl in corr_keys])
            for idx, (_, lbl) in enumerate(corr_keys):
                c_writer.writerow([lbl] + [f"{corr_mat[idx, j]:.4f}" for j in range(len(corr_keys))])

        # Plot Correlation Heatmap
        plt.figure(figsize=(9, 7.5))
        im = plt.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson Correlation")
        labels = [lbl for _, lbl in corr_keys]
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
        plt.yticks(range(len(labels)), labels, fontsize=9)
        plt.title("Avenue Normal Training Set: Feature Correlation Matrix", fontsize=12, fontweight="bold")

        for r_idx in range(len(labels)):
            for c_idx in range(len(labels)):
                val = corr_mat[r_idx, c_idx]
                color_txt = "white" if abs(val) > 0.55 else "black"
                plt.text(c_idx, r_idx, f"{val:.2f}", ha="center", va="center", color=color_txt, fontsize=8)

        plt.tight_layout()
        plt.savefig(corr_img_path, dpi=150)
        plt.close()

    # --------------------------------------------------------------------------
    # Step K: Dataset Metadata JSON
    # --------------------------------------------------------------------------
    all_speeds = [float(r["cluster_speed"]) for r in all_merged_feature_rows if float(r["cluster_speed"]) > 0]
    all_sizes = [int(r["number_of_people"]) for r in all_merged_feature_rows]
    all_densities = [float(r["density"]) for r in all_merged_feature_rows]
    all_coherences = [float(r["directional_coherence"]) for r in all_merged_feature_rows]
    all_durations = [int(r["duration_frames"]) for r in all_merged_summary_rows]

    mean_size = float(np.mean(all_sizes)) if all_sizes else 0.0
    med_size = float(np.median(all_sizes)) if all_sizes else 0.0
    max_size = int(np.max(all_sizes)) if all_sizes else 0

    mean_spd = float(np.mean(all_speeds)) if all_speeds else 0.0
    med_spd = float(np.median(all_speeds)) if all_speeds else 0.0
    max_spd = float(np.max(all_speeds)) if all_speeds else 0.0

    mean_dens = float(np.mean(all_densities)) if all_densities else 0.0
    mean_coh = float(np.mean(all_coherences)) if all_coherences else 0.0
    mean_dur = float(np.mean(all_durations)) if all_durations else 0.0

    dataset_metadata = {
        "dataset_name": "Avenue Normal Training Spatiotemporal Feature Baseline",
        "created_at": datetime.now().isoformat(),
        "total_training_videos": len(video_dirs),
        "total_frames_processed": processed_frames_all,
        "total_person_detections": total_detections_all,
        "total_unique_person_tracks": unique_tracks_all,
        "total_persistent_clusters": len(all_merged_summary_rows),
        "total_feature_rows": len(all_merged_feature_rows),
        "configuration": config_dict,
        "quality_metrics": {
            "nan_values": nan_count,
            "inf_values": inf_count,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
        },
        "aggregate_statistics": {
            "mean_cluster_size": mean_size,
            "median_cluster_size": med_size,
            "max_cluster_size": max_size,
            "mean_cluster_speed": mean_spd,
            "median_cluster_speed": med_spd,
            "max_cluster_speed": max_spd,
            "mean_density": mean_dens,
            "mean_directional_coherence": mean_coh,
            "mean_cluster_duration_frames": mean_dur,
        },
        "output_files": {
            "merged_features_csv": str(merged_features_csv),
            "merged_summary_csv": str(merged_summary_csv),
            "distributions_dir": str(distributions_dir),
            "correlation_image": str(corr_img_path),
            "correlation_csv": str(corr_csv_path),
        },
    }

    with open(dataset_metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    return dataset_metadata


# ==============================================================================
# MAIN BATCH DRIVER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Construct full Avenue normal training spatiotemporal feature dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/Avenue_Dataset/Avenue Dataset/training_videos",
        help="Path to Avenue training videos directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/training_dataset",
        help="Directory to save per-video and merged training datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model variant or weights path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO detection confidence threshold (default: 0.15).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=60.0,
        help="DBSCAN spatial neighborhood radius in pixels (default: 60.0).",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=3,
        help="DBSCAN minimum members for a cluster (default: 3).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process a specific video file (e.g. '01.avi' or full path).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process per video (None for full video).",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=1,
        help="Starting frame index (default: 1).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Ending frame index (default: None).",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Save annotated MP4 video output for each video (default: False).",
    )
    parser.add_argument(
        "--benchmark_models",
        action="store_true",
        default=False,
        help="Run short benchmark across YOLO model variants on representative frames.",
    )

    args = parser.parse_args()

    # Discover videos
    train_dir = Path(args.dataset_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training dataset directory not found: {train_dir}")

    all_vids = sorted(list(train_dir.glob("*.avi")))
    if not all_vids:
        raise FileNotFoundError(f"No AVI videos found in: {train_dir}")

    # Optional model benchmark
    if args.benchmark_models:
        rep_video = all_vids[0]
        run_model_benchmark(
            video_path=rep_video,
            models=["yolov8n.pt", "yolov8x.pt"],
            num_frames=100,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        return

    # Select target video(s)
    if args.video:
        target_name = Path(args.video).name
        target_vids = [v for v in all_vids if v.name == target_name or v.stem == Path(args.video).stem]
        if not target_vids:
            # Check direct path
            if Path(args.video).exists():
                target_vids = [Path(args.video)]
            else:
                raise FileNotFoundError(f"Target video '{args.video}' not found in training dataset.")
    else:
        target_vids = all_vids

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": args.model,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "max_frames": args.max_frames,
        "save_video": args.save_video,
    }

    print("=" * 70)
    print("AVENUE NORMAL TRAINING DATASET CONSTRUCTION PIPELINE")
    print("=" * 70)
    print(f"Target videos to process:        {len(target_vids)}")
    print(f"YOLO Model weights:              {args.model}")
    print(f"Confidence threshold:            {args.conf}")
    print(f"Inference image size:            {args.imgsz}")
    print(f"DBSCAN spatial radius (eps):     {args.dbscan_eps} px")
    print(f"DBSCAN min samples:              {args.dbscan_min_samples}")
    print(f"Output directory:                {output_base}")
    print(f"Frame processing scope:          {'Full video' if args.max_frames is None else f'First {args.max_frames} frames'}")
    print(f"Save annotated videos:           {args.save_video}")
    print("=" * 70 + "\n")

    # Load YOLO Model once for batch processing
    print(f"Loading YOLO model [{args.model}] into memory...")
    yolo_model = YOLO(args.model)
    print("YOLO model ready.\n")

    successful_videos = []
    failed_videos = []
    t_batch_start = time.time()

    for idx, vid_path in enumerate(target_vids, 1):
        v_stem = vid_path.stem  # e.g. '01'
        v_folder_name = f"video_{v_stem}"
        v_out_dir = output_base / v_folder_name

        print(f"[{idx}/{len(target_vids)}] Processing {vid_path.name} -> {v_folder_name}/ ...")
        t_v0 = time.time()

        try:
            v_meta = process_single_video(
                video_path=vid_path,
                output_video_dir=v_out_dir,
                model=yolo_model,
                conf=args.conf,
                imgsz=args.imgsz,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                save_video=args.save_video,
            )
            v_elapsed = time.time() - t_v0
            successful_videos.append(vid_path.name)
            print(f"    Done in {v_elapsed:.1f}s ({v_meta['processed_frames']} frames | {v_meta['persistent_clusters_count']} clusters | {v_meta['cluster_feature_rows_count']} rows)")
        except Exception as e:
            failed_videos.append({"video": vid_path.name, "error": str(e)})
            print(f"    ERROR processing {vid_path.name}: {e}")

    # Merge and analyze across all processed videos
    dataset_metadata = merge_and_analyze_dataset(output_base, config_dict)

    # --------------------------------------------------------------------------
    # Final Standardized Summary Block
    # --------------------------------------------------------------------------
    agg = dataset_metadata["aggregate_statistics"]

    print("\n" + "=" * 52)
    print("AVENUE NORMAL TRAINING DATASET SUMMARY")
    print("=" * 52)
    print(f"Training videos:             {len(target_vids)}")
    print(f"Successful videos:           {len(successful_videos)}")
    print(f"Failed videos:               {len(failed_videos)}")
    print("")
    print(f"Total frames:                {dataset_metadata['total_frames_processed']}")
    print(f"Total person tracks:         {dataset_metadata['total_unique_person_tracks']}")
    print(f"Total persistent clusters:   {dataset_metadata['total_persistent_clusters']}")
    print(f"Total cluster feature rows:  {dataset_metadata['total_feature_rows']}")
    print("")
    print(f"Mean cluster size:           {agg['mean_cluster_size']:.2f}")
    print(f"Median cluster size:         {agg['median_cluster_size']:.2f}")
    print(f"Maximum cluster size:        {agg['max_cluster_size']}")
    print("")
    print(f"Mean cluster speed:          {agg['mean_cluster_speed']:.2f} px/s")
    print(f"Median cluster speed:        {agg['median_cluster_speed']:.2f} px/s")
    print(f"Maximum cluster speed:       {agg['max_cluster_speed']:.2f} px/s")
    print("")
    print(f"Mean density:                {agg['mean_density']:.6f} people/px^2")
    print(f"Mean directional coherence:  {agg['mean_directional_coherence']:.4f}")
    print("")
    print(f"Final merged feature CSV:    {dataset_metadata['output_files']['merged_features_csv']}")
    print(f"Final cluster summary CSV:   {dataset_metadata['output_files']['merged_summary_csv']}")
    print(f"Metadata JSON:               {output_base / 'normal_dataset_metadata.json'}")
    print("=" * 52)


if __name__ == "__main__":
    main()
