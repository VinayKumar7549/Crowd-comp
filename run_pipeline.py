import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import time
import yaml
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
import torch
from ultralytics import YOLO


def wrap_angle_deg(angle_diff: float) -> float:
    """Wrap angular difference into range [-180, 180] degrees."""
    return (angle_diff + 180.0) % 360.0 - 180.0


def get_persistent_color(persistent_id: int | str) -> tuple[int, int, int]:
    """Generate a consistent, distinct BGR color for each persistent cluster ID."""
    palette = [
        (255, 120, 0),    # Vivid Blue (BGR)
        (0, 200, 50),     # Green
        (0, 140, 255),    # Orange
        (220, 50, 220),   # Magenta
        (0, 230, 255),    # Yellow
        (255, 0, 120),    # Indigo/Purple
        (40, 100, 255),   # Red-Orange
        (180, 220, 0),    # Teal
        (120, 180, 255),  # Coral
        (200, 150, 50),   # Steel Blue
    ]
    if isinstance(persistent_id, str):
        h = abs(hash(persistent_id))
        return palette[h % len(palette)]
    return palette[int(persistent_id) % len(palette)]


def get_small_group_color(sg_id: int | str) -> tuple[int, int, int]:
    """Generate distinct BGR color for persistent small groups."""
    sg_palette = [
        (0, 180, 255),    # Warm Amber
        (255, 200, 0),    # Electric Cyan
        (180, 100, 255),  # Soft Pink / Violet
        (50, 220, 150),   # Mint Green
        (200, 160, 0),    # Bright Sky Blue
    ]
    if isinstance(sg_id, str):
        h = abs(hash(sg_id))
        return sg_palette[h % len(sg_palette)]
    return sg_palette[int(sg_id) % len(sg_palette)]


def natural_sort_key(s: Path) -> list:
    """Sort path filenames numerically for proper frame order (e.g. seq_000002.jpg after seq_000001.jpg)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s.name)]


class FrameStreamer:
    """Discovers and streams frames sequentially from either a video file or an image frame directory."""

    def __init__(self, input_path: str | Path, start_frame: int = 1, max_frames: int | None = None, target_fps: float = 25.0):
        self.input_path = Path(input_path)
        self.start_frame = max(1, start_frame)
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.is_video = False
        self.is_dir = False
        self.frame_files: list[Path] = []
        self.cap: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = target_fps

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        if self.input_path.is_dir():
            self.is_dir = True
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            all_files = [f for f in self.input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            all_files.sort(key=natural_sort_key)
            if not all_files:
                raise ValueError(f"No supported image files found in directory: {self.input_path}")
            
            # Read first image to determine width and height
            sample_img = cv2.imread(str(all_files[0]))
            if sample_img is None:
                raise ValueError(f"Unable to read image: {all_files[0]}")
            self.height, self.width = sample_img.shape[:2]
            
            # Slice frame files according to start_frame and max_frames
            start_idx = self.start_frame - 1
            if start_idx >= len(all_files):
                raise ValueError(f"start_frame ({self.start_frame}) exceeds available frame count ({len(all_files)})")
            
            if self.max_frames is not None:
                self.frame_files = all_files[start_idx : start_idx + self.max_frames]
            else:
                self.frame_files = all_files[start_idx:]
            self.total_frames = len(self.frame_files)

        elif self.input_path.is_file():
            self.is_video = True
            self.cap = cv2.VideoCapture(str(self.input_path))
            if not self.cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.input_path}")
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if cap_fps > 0 and cap_fps == cap_fps:
                self.fps = cap_fps
            
            total_vid_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = total_vid_frames if total_vid_frames > 0 else 0
        else:
            raise ValueError(f"Unsupported input path type: {self.input_path}")

    def stream(_self):
        """Generator that yields (frame_number, BGR_image_array)."""
        if _self.is_dir:
            for idx, fpath in enumerate(_self.frame_files, start=_self.start_frame):
                frame = cv2.imread(str(fpath))
                if frame is not None:
                    yield idx, frame
        elif _self.is_video and _self.cap is not None:
            curr_frame = 0
            yielded_count = 0
            while True:
                ret, frame = _self.cap.read()
                if not ret or frame is None:
                    break
                curr_frame += 1
                if curr_frame < _self.start_frame:
                    continue
                yielded_count += 1
                yield curr_frame, frame
                if _self.max_frames is not None and yielded_count >= _self.max_frames:
                    break

    def close(self):
        if self.cap is not None:
            self.cap.release()


def run_pipeline(
    input_path: str,
    output_path: str,
    model_path: str = "yolov8n.pt",
    device_arg: str = "auto",
    conf: float = 0.15,
    imgsz: int = 640,
    start_frame: int = 1,
    max_frames: int | None = None,
    fps_arg: float = 25.0,
    use_half: bool = False,
    eps: float = 60.0,
    min_samples: int = 3,
    small_group_eps: float = 60.0,
    max_centroid_dist: float = 80.0,
    max_gap_frames: int = 5,
    use_sahi: bool = False,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.20,
    overlap_width_ratio: float = 0.20,
    highlight_stationary: bool = False,
    stationary_threshold: float = 5.0,
    save_features: bool = True,
    preloaded_model: YOLO | None = None,
):
    streamer = FrameStreamer(input_path, start_frame=start_frame, max_frames=max_frames, target_fps=fps_arg)

    # Resolve compute device
    if device_arg.lower() == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_arg

    cuda_active = torch.cuda.is_available() and ("cuda" in device_str.lower())
    device_display = f"GPU ({torch.cuda.get_device_name(0)})" if cuda_active else "CPU"
    fp16_mode = use_half and cuda_active

    # Prepare output directory & VideoWriter
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, streamer.fps, (streamer.width, streamer.height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_file), fourcc, streamer.fps, (streamer.width, streamer.height))

    # Load model / SAHI detector ONCE
    if use_sahi:
        import ultralytics
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from ultralytics.engine.results import Results
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            device=device_str,
        )
        pkg_dir = os.path.dirname(ultralytics.__file__)
        yaml_path = os.path.join(pkg_dir, "cfg", "trackers", "bytetrack.yaml")
        with open(yaml_path, "r") as f:
            tracker_cfg = IterableSimpleNamespace(**yaml.safe_load(f))
        tracker = BYTETracker(tracker_cfg)
        model = None
    else:
        model = preloaded_model if preloaded_model is not None else YOLO(model_path)
        sahi_model = None
        tracker = None

    # Persistent group tracking states
    active_crowd_tracks: dict[int, dict] = {}
    next_persistent_crowd_id = 0

    active_sg_tracks: dict[int, dict] = {}
    next_persistent_sg_id = 0

    track_history: dict[int, list[tuple[float, float]]] = {}

    all_tracking_rows: list[dict] = []
    tracks_by_id: dict[int, list[dict]] = defaultdict(list)
    all_person_cluster_rows: list[dict] = []
    all_cluster_feature_rows: list[dict] = []
    all_persistent_cluster_rows: list[dict] = []
    persistent_cluster_obs: dict[int, list[dict]] = defaultdict(list)
    cluster_member_tracks: dict[tuple[int, int], list[int]] = defaultdict(list)

    processed_frames_count = 0
    total_raw_detected_persons = 0
    raw_detected_list = []
    total_tracked_persons = 0
    total_crowd_clusters = 0
    total_small_groups = 0
    total_isolated_persons = 0
    t_start = time.time()

    active_imgsz = imgsz

    try:
        for f_idx, frame in streamer.stream():
            processed_frames_count += 1

            # Step 1: Person Detection + ByteTrack
            if use_sahi:
                sahi_res = get_sliced_prediction(
                    frame,
                    sahi_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    verbose=0,
                )
                person_preds = [p for p in sahi_res.object_prediction_list if p.category.id == 0]
                raw_n = len(person_preds)
                total_raw_detected_persons += raw_n
                raw_detected_list.append(raw_n)

                if person_preds:
                    boxes_list = [
                        [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy, p.score.value, 0.0]
                        for p in person_preds
                    ]
                    boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
                else:
                    boxes_tensor = torch.zeros((0, 6), dtype=torch.float32)

                results_obj = Results(orig_img=frame, path="", names={0: "person"}, boxes=boxes_tensor)
                tracked = tracker.update(results_obj.boxes, img=frame)

                people_detections = []
                if len(tracked) > 0:
                    for row in tracked:
                        x1, y1, x2, y2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
                        people_detections.append({
                            "track_id": tid,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "foot_x": (x1 + x2) / 2.0,
                            "foot_y": y2,
                        })
            else:
                track_kwargs = dict(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=conf,
                    imgsz=active_imgsz,
                    verbose=False,
                    device=device_str,
                )
                if fp16_mode:
                    track_kwargs["half"] = True

                try:
                    results = model.track(**track_kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("out of memory" in err_msg or "cudaoutofmemoryerror" in err_msg) and active_imgsz > 640:
                        print(f"[WARNING] CUDA Out of Memory with imgsz={active_imgsz}. Automatically falling back to imgsz=640.")
                        if cuda_active:
                            torch.cuda.empty_cache()
                        active_imgsz = 640
                        track_kwargs["imgsz"] = 640
                        results = model.track(**track_kwargs)
                    else:
                        raise e

                boxes = results[0].boxes
                raw_n = len(boxes) if (boxes is not None and len(boxes) > 0) else 0
                total_raw_detected_persons += raw_n
                raw_detected_list.append(raw_n)

                people_detections = []
                if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                    tids = boxes.id.int().cpu().tolist()
                    xyxys = boxes.xyxy.cpu().tolist()
                    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(tids)
                    for tid, conf_val, (x1, y1, x2, y2) in zip(tids, confs, xyxys):
                        people_detections.append({
                            "track_id": int(tid),
                            "conf": float(conf_val),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "foot_x": float((x1 + x2) / 2.0),
                            "foot_y": float(y2),
                        })

            for p in people_detections:
                tid = p["track_id"]
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((p["foot_x"], p["foot_y"]))
                if save_features:
                    obs = {
                        "frame": f_idx,
                        "track_id": tid,
                        "x1": p["x1"],
                        "y1": p["y1"],
                        "x2": p["x2"],
                        "y2": p["y2"],
                        "width": p["x2"] - p["x1"],
                        "height": p["y2"] - p["y1"],
                        "center_x": (p["x1"] + p["x2"]) / 2.0,
                        "center_y": (p["y1"] + p["y2"]) / 2.0,
                        "foot_x": p["foot_x"],
                        "foot_y": p["foot_y"],
                        "confidence": p.get("conf", 1.0),
                    }
                    all_tracking_rows.append(obs)
                    tracks_by_id[tid].append(obs)

            n_people = len(people_detections)
            total_tracked_persons += n_people

            # Step 2: Spatial DBSCAN & Group Disambiguation
            if n_people < min_samples:
                db_labels = np.full(n_people, -1, dtype=int)
            else:
                coords = np.array([[p["foot_x"], p["foot_y"]] for p in people_detections])
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                db_labels = db.labels_

            curr_crowd_clusters = []
            unique_cids = sorted([c for c in set(db_labels) if c != -1])
            for cid in unique_cids:
                members = [p for p, l in zip(people_detections, db_labels) if l == cid]
                foot_xs = [m["foot_x"] for m in members]
                foot_ys = [m["foot_y"] for m in members]
                member_ids = {m["track_id"] for m in members}
                curr_crowd_clusters.append({
                    "group_type": "crowd_cluster",
                    "dbscan_cluster_id": cid,
                    "number_of_people": len(members),
                    "centroid_x": float(np.mean(foot_xs)),
                    "centroid_y": float(np.mean(foot_ys)),
                    "cluster_x1": float(min(m["x1"] for m in members)),
                    "cluster_y1": float(min(m["y1"] for m in members)),
                    "cluster_x2": float(max(m["x2"] for m in members)),
                    "cluster_y2": float(max(m["y2"] for m in members)),
                    "members": members,
                    "member_ids": member_ids,
                })

            total_crowd_clusters += len(curr_crowd_clusters)

            # Process Noise Points -> Small Groups or Isolated Persons
            noise_indices = [i for i, l in enumerate(db_labels) if l == -1]
            curr_small_groups = []
            isolated_persons = []

            if len(noise_indices) >= 2:
                noise_people = [people_detections[i] for i in noise_indices]
                noise_coords = np.array([[p["foot_x"], p["foot_y"]] for p in noise_people])
                sg_db = DBSCAN(eps=small_group_eps, min_samples=2).fit(noise_coords)
                sg_labels = sg_db.labels_

                unique_sg_ids = sorted([sg for sg in set(sg_labels) if sg != -1])
                for sg_id in unique_sg_ids:
                    sg_members = [p for p, l in zip(noise_people, sg_labels) if l == sg_id]
                    foot_xs = [m["foot_x"] for m in sg_members]
                    foot_ys = [m["foot_y"] for m in sg_members]
                    sg_member_ids = {m["track_id"] for m in sg_members}
                    curr_small_groups.append({
                        "group_type": "small_group",
                        "number_of_people": len(sg_members),
                        "centroid_x": float(np.mean(foot_xs)),
                        "centroid_y": float(np.mean(foot_ys)),
                        "cluster_x1": float(min(m["x1"] for m in sg_members)),
                        "cluster_y1": float(min(m["y1"] for m in sg_members)),
                        "cluster_x2": float(max(m["x2"] for m in sg_members)),
                        "cluster_y2": float(max(m["y2"] for m in sg_members)),
                        "members": sg_members,
                        "member_ids": sg_member_ids,
                    })

                isolated_indices = [i for i, l in enumerate(sg_labels) if l == -1]
                isolated_persons = [noise_people[i] for i in isolated_indices]
            else:
                isolated_persons = [people_detections[i] for i in noise_indices]

            total_small_groups += len(curr_small_groups)
            total_isolated_persons += len(isolated_persons)

            # Step 3: Persistent Crowd Cluster Tracking
            if not active_crowd_tracks and curr_crowd_clusters:
                for c in curr_crowd_clusters:
                    pid = next_persistent_crowd_id
                    next_persistent_crowd_id += 1
                    active_crowd_tracks[pid] = {
                        "persistent_id": pid,
                        "centroid_x": c["centroid_x"],
                        "centroid_y": c["centroid_y"],
                        "member_ids": c["member_ids"],
                        "missed_frames": 0,
                    }
                    c["persistent_id"] = pid
            elif not curr_crowd_clusters:
                for pid, trk in list(active_crowd_tracks.items()):
                    trk["missed_frames"] += 1
                    if trk["missed_frames"] > max_gap_frames:
                        del active_crowd_tracks[pid]
            else:
                active_pids = list(active_crowd_tracks.keys())
                cost_matrix = np.full((len(active_pids), len(curr_crowd_clusters)), 1e6, dtype=float)

                for i, pid in enumerate(active_pids):
                    trk = active_crowd_tracks[pid]
                    for j, c in enumerate(curr_crowd_clusters):
                        dx = c["centroid_x"] - trk["centroid_x"]
                        dy = c["centroid_y"] - trk["centroid_y"]
                        dist = math.sqrt(dx * dx + dy * dy)

                        union_m = trk["member_ids"] | c["member_ids"]
                        overlap = (len(trk["member_ids"] & c["member_ids"]) / len(union_m)) if union_m else 0.0
                        cost = dist * (1.0 - 0.5 * overlap)

                        if dist <= max_centroid_dist or (overlap >= 0.20 and dist <= max_centroid_dist * 1.5):
                            cost_matrix[i, j] = cost

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_active = set()
                matched_curr = set()

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1e5:
                        pid = active_pids[r]
                        curr_c = curr_crowd_clusters[c]
                        curr_c["persistent_id"] = pid
                        active_crowd_tracks[pid]["centroid_x"] = curr_c["centroid_x"]
                        active_crowd_tracks[pid]["centroid_y"] = curr_c["centroid_y"]
                        active_crowd_tracks[pid]["member_ids"] = curr_c["member_ids"]
                        active_crowd_tracks[pid]["missed_frames"] = 0
                        matched_active.add(pid)
                        matched_curr.add(c)

                for pid in active_pids:
                    if pid not in matched_active:
                        active_crowd_tracks[pid]["missed_frames"] += 1
                        if active_crowd_tracks[pid]["missed_frames"] > max_gap_frames:
                            del active_crowd_tracks[pid]

                for j, c in enumerate(curr_crowd_clusters):
                    if j not in matched_curr:
                        pid = next_persistent_crowd_id
                        next_persistent_crowd_id += 1
                        active_crowd_tracks[pid] = {
                            "persistent_id": pid,
                            "centroid_x": c["centroid_x"],
                            "centroid_y": c["centroid_y"],
                            "member_ids": c["member_ids"],
                            "missed_frames": 0,
                        }
                        c["persistent_id"] = pid

            # Step 4: Persistent Small Group Tracking
            if not active_sg_tracks and curr_small_groups:
                for sg in curr_small_groups:
                    psg_id = next_persistent_sg_id
                    next_persistent_sg_id += 1
                    active_sg_tracks[psg_id] = {
                        "persistent_sg_id": psg_id,
                        "centroid_x": sg["centroid_x"],
                        "centroid_y": sg["centroid_y"],
                        "member_ids": sg["member_ids"],
                        "missed_frames": 0,
                    }
                    sg["persistent_sg_id"] = psg_id
            elif not curr_small_groups:
                for psg_id, trk in list(active_sg_tracks.items()):
                    trk["missed_frames"] += 1
                    if trk["missed_frames"] > max_gap_frames:
                        del active_sg_tracks[psg_id]
            else:
                active_sg_ids = list(active_sg_tracks.keys())
                cost_mat = np.full((len(active_sg_ids), len(curr_small_groups)), 1e6, dtype=float)

                for i, psg_id in enumerate(active_sg_ids):
                    trk = active_sg_tracks[psg_id]
                    for j, sg in enumerate(curr_small_groups):
                        dx = sg["centroid_x"] - trk["centroid_x"]
                        dy = sg["centroid_y"] - trk["centroid_y"]
                        dist = math.sqrt(dx * dx + dy * dy)

                        union_m = trk["member_ids"] | sg["member_ids"]
                        overlap = (len(trk["member_ids"] & sg["member_ids"]) / len(union_m)) if union_m else 0.0
                        cost = dist * (1.0 - 0.5 * overlap)

                        if dist <= max_centroid_dist or (overlap >= 0.30 and dist <= max_centroid_dist * 1.5):
                            cost_mat[i, j] = cost

                row_ind, col_ind = linear_sum_assignment(cost_mat)
                matched_act_sg = set()
                matched_cur_sg = set()

                for r, c in zip(row_ind, col_ind):
                    if cost_mat[r, c] < 1e5:
                        psg_id = active_sg_ids[r]
                        curr_sg = curr_small_groups[c]
                        curr_sg["persistent_sg_id"] = psg_id
                        active_sg_tracks[psg_id]["centroid_x"] = curr_sg["centroid_x"]
                        active_sg_tracks[psg_id]["centroid_y"] = curr_sg["centroid_y"]
                        active_sg_tracks[psg_id]["member_ids"] = curr_sg["member_ids"]
                        active_sg_tracks[psg_id]["missed_frames"] = 0
                        matched_act_sg.add(psg_id)
                        matched_cur_sg.add(c)

                for psg_id in active_sg_ids:
                    if psg_id not in matched_act_sg:
                        active_sg_tracks[psg_id]["missed_frames"] += 1
                        if active_sg_tracks[psg_id]["missed_frames"] > max_gap_frames:
                            del active_sg_tracks[psg_id]

                for j, sg in enumerate(curr_small_groups):
                    if j not in matched_cur_sg:
                        psg_id = next_persistent_sg_id
                        next_persistent_sg_id += 1
                        active_sg_tracks[psg_id] = {
                            "persistent_sg_id": psg_id,
                            "centroid_x": sg["centroid_x"],
                            "centroid_y": sg["centroid_y"],
                            "member_ids": sg["member_ids"],
                            "missed_frames": 0,
                        }
                        sg["persistent_sg_id"] = psg_id

            if save_features:
                # Record crowd clusters
                for c in curr_crowd_clusters:
                    pid = c["persistent_id"]
                    cid = c["dbscan_cluster_id"]
                    n_p = c["number_of_people"]
                    members = c["members"]
                    cx1, cy1 = c["cluster_x1"], c["cluster_y1"]
                    cx2, cy2 = c["cluster_x2"], c["cluster_y2"]
                    w = cx2 - cx1
                    h = cy2 - cy1
                    area = w * h
                    density = (n_p / area) if area > 0 else 0.0
                    min_x = min(m["foot_x"] for m in members)
                    max_x = max(m["foot_x"] for m in members)
                    min_y = min(m["foot_y"] for m in members)
                    max_y = max(m["foot_y"] for m in members)

                    c_dict = {
                        "frame": f_idx,
                        "group_type": "crowd_cluster",
                        "group_id": pid,
                        "persistent_cluster_id": pid,
                        "dbscan_cluster_id": cid,
                        "number_of_people": n_p,
                        "centroid_x": c["centroid_x"],
                        "centroid_y": c["centroid_y"],
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "cluster_x1": cx1,
                        "cluster_y1": cy1,
                        "cluster_x2": cx2,
                        "cluster_y2": cy2,
                        "width": w,
                        "height": h,
                        "area": area,
                        "density": density,
                    }
                    all_cluster_feature_rows.append(c_dict)
                    all_persistent_cluster_rows.append(c_dict)
                    persistent_cluster_obs[pid].append(c_dict)
                    cluster_member_tracks[(f_idx, pid)] = [m["track_id"] for m in members]

                    for m in members:
                        all_person_cluster_rows.append({
                            "frame": f_idx,
                            "track_id": m["track_id"],
                            "foot_x": m["foot_x"],
                            "foot_y": m["foot_y"],
                            "cluster_id": cid,
                            "group_type": "crowd_cluster",
                            "group_id": pid,
                            "group_size": n_p,
                            "cluster_x1": cx1,
                            "cluster_y1": cy1,
                            "cluster_x2": cx2,
                            "cluster_y2": cy2,
                        })

                # Record small groups
                for sg in curr_small_groups:
                    psg_id = sg["persistent_sg_id"]
                    n_sg = sg["number_of_people"]
                    sg_members = sg["members"]
                    sg_x1, sg_y1 = sg["cluster_x1"], sg["cluster_y1"]
                    sg_x2, sg_y2 = sg["cluster_x2"], sg["cluster_y2"]
                    w = sg_x2 - sg_x1
                    h = sg_y2 - sg_y1
                    area = w * h
                    density = (n_sg / area) if area > 0 else 0.0
                    min_x = min(m["foot_x"] for m in sg_members)
                    max_x = max(m["foot_x"] for m in sg_members)
                    min_y = min(m["foot_y"] for m in sg_members)
                    max_y = max(m["foot_y"] for m in sg_members)

                    sg_dict = {
                        "frame": f_idx,
                        "group_type": "small_group",
                        "group_id": f"SG_{psg_id}",
                        "persistent_cluster_id": f"SG_{psg_id}",
                        "dbscan_cluster_id": -1,
                        "number_of_people": n_sg,
                        "centroid_x": sg["centroid_x"],
                        "centroid_y": sg["centroid_y"],
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "cluster_x1": sg_x1,
                        "cluster_y1": sg_y1,
                        "cluster_x2": sg_x2,
                        "cluster_y2": sg_y2,
                        "width": w,
                        "height": h,
                        "area": area,
                        "density": density,
                    }
                    all_cluster_feature_rows.append(sg_dict)

                    for m in sg_members:
                        all_person_cluster_rows.append({
                            "frame": f_idx,
                            "track_id": m["track_id"],
                            "foot_x": m["foot_x"],
                            "foot_y": m["foot_y"],
                            "cluster_id": -1,
                            "group_type": "small_group",
                            "group_id": f"SG_{psg_id}",
                            "group_size": n_sg,
                            "cluster_x1": sg_x1,
                            "cluster_y1": sg_y1,
                            "cluster_x2": sg_x2,
                            "cluster_y2": sg_y2,
                        })

                # Record isolated persons
                for p in isolated_persons:
                    all_person_cluster_rows.append({
                        "frame": f_idx,
                        "track_id": p["track_id"],
                        "foot_x": p["foot_x"],
                        "foot_y": p["foot_y"],
                        "cluster_id": -1,
                        "group_type": "isolated_person",
                        "group_id": p["track_id"],
                        "group_size": 1,
                        "cluster_x1": p["x1"],
                        "cluster_y1": p["y1"],
                        "cluster_x2": p["x2"],
                        "cluster_y2": p["y2"],
                    })

            # Step 5: Visual Overlay Rendering
            vid_h, vid_w = streamer.height, streamer.width

            # 1. Crowd Clusters (3+ people)
            for c in curr_crowd_clusters:
                pid = c["persistent_id"]
                color = get_persistent_color(pid)
                pad = 6
                bx1 = max(0, int(c["cluster_x1"]) - pad)
                by1 = max(0, int(c["cluster_y1"]) - pad)
                bx2 = min(vid_w - 1, int(c["cluster_x2"]) + pad)
                by2 = min(vid_h - 1, int(c["cluster_y2"]) + pad)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cx, cy = int(c["centroid_x"]), int(c["centroid_y"])
                cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

                label = f"Group {pid} | People={c['number_of_people']}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (bx1 + 3, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 2. Small Groups (2 people)
            for sg in curr_small_groups:
                psg_id = sg["persistent_sg_id"]
                color = get_small_group_color(psg_id)
                pad = 4
                bx1 = max(0, int(sg["cluster_x1"]) - pad)
                by1 = max(0, int(sg["cluster_y1"]) - pad)
                bx2 = min(vid_w - 1, int(sg["cluster_x2"]) + pad)
                by2 = min(vid_h - 1, int(sg["cluster_y2"]) + pad)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cx, cy = int(sg["centroid_x"]), int(sg["centroid_y"])
                cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_DIAMOND, 10, 1)

                label = f"Small Group {psg_id} | N={sg['number_of_people']}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (bx1 + 3, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

            # 3. Isolated Persons
            for p in isolated_persons:
                color = (180, 180, 180)
                px1 = max(0, int(p["x1"]))
                py1 = max(0, int(p["y1"]))
                px2 = min(vid_w - 1, int(p["x2"]))
                py2 = min(vid_h - 1, int(p["y2"]))

                label = f"Person {p['track_id']}"
                if highlight_stationary:
                    pts = track_history.get(p['track_id'], [])
                    if len(pts) > 0:
                        disp = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
                        if disp < stationary_threshold:
                            label = f"NEAR-STATIONARY ID {p['track_id']}"
                            color = (0, 0, 230)

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1 if not (highlight_stationary and "NEAR-STATIONARY" in label) else 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                bg_color = (0, 0, 180) if (highlight_stationary and "NEAR-STATIONARY" in label) else (50, 50, 50)
                cv2.rectangle(frame, (px1, max(0, py1 - th - 4)), (px1 + tw + 4, py1), bg_color, -1)
                cv2.putText(
                    frame,
                    label,
                    (px1 + 2, py1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255) if (highlight_stationary and "NEAR-STATIONARY" in label) else (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

            # Highlight near-stationary tracks in clusters/groups if requested
            if highlight_stationary:
                for p in people_detections:
                    pts = track_history.get(p["track_id"], [])
                    if len(pts) > 0:
                        disp = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
                        if disp < stationary_threshold:
                            px1, py1 = max(0, int(p["x1"])), max(0, int(p["y1"]))
                            label = f"NEAR-STATIONARY ID {p['track_id']}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                            cv2.rectangle(frame, (px1, max(0, py1 - th - 4)), (px1 + tw + 4, py1), (0, 0, 180), -1)
                            cv2.putText(frame, label, (px1 + 2, py1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Foot Points
            for c in curr_crowd_clusters:
                color = get_persistent_color(c["persistent_id"])
                for m in c["members"]:
                    fx, fy = int(round(m["foot_x"])), int(round(m["foot_y"]))
                    cv2.circle(frame, (fx, fy), 4, color, -1)
                    cv2.circle(frame, (fx, fy), 5, (255, 255, 255), 1)

            for sg in curr_small_groups:
                color = get_small_group_color(sg["persistent_sg_id"])
                for m in sg["members"]:
                    fx, fy = int(round(m["foot_x"])), int(round(m["foot_y"]))
                    cv2.circle(frame, (fx, fy), 4, color, -1)
                    cv2.circle(frame, (fx, fy), 5, (0, 0, 0), 1)

            for p in isolated_persons:
                fx, fy = int(round(p["foot_x"])), int(round(p["foot_y"]))
                cv2.circle(frame, (fx, fy), 3, (180, 180, 180), -1)
                cv2.drawMarker(frame, (fx, fy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 4, 1)

            # Write frame to video
            writer.write(frame)

    finally:
        streamer.close()
        writer.release()

    t_elapsed = time.time() - t_start
    proc_fps = (processed_frames_count / t_elapsed) if t_elapsed > 0 else 0.0
    avg_raw_detected = (total_raw_detected_persons / processed_frames_count) if processed_frames_count > 0 else 0.0
    avg_tracked_persons = (total_tracked_persons / processed_frames_count) if processed_frames_count > 0 else 0.0
    min_raw_detected = min(raw_detected_list) if raw_detected_list else 0
    max_raw_detected = max(raw_detected_list) if raw_detected_list else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_active else "None"
    duration_sec = (processed_frames_count / streamer.fps) if streamer.fps > 0 else 0.0
    end_f = start_frame + processed_frames_count - 1

    print("Input:", input_path)
    print(f"Frame range: {start_frame} to {end_f}")
    print("Frames processed:", processed_frames_count)
    print(f"Average detected persons/frame: {avg_raw_detected:.2f}")
    print(f"Average active tracks/frame: {avg_tracked_persons:.2f}")
    print(f"Approximate number of crowd-group observations: {total_crowd_clusters}")
    print("CUDA:", "Yes" if cuda_active else "No")
    print("GPU:", gpu_name)
    print(f"Processing FPS: {proc_fps:.2f}")
    print(f"Output FPS: {streamer.fps:.1f}")
    print(f"Output duration: {duration_sec:.1f} seconds")
    print("Output path:", output_path)

    if highlight_stationary:
        total_tracks_count = len(track_history)
        moving_tracks = []
        stationary_tracks = []
        mannequin_stationary_tracks = []

        for tid, pts in track_history.items():
            disp = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
            initial_x, initial_y = pts[0]
            if disp < stationary_threshold:
                stationary_tracks.append((tid, disp, initial_x, initial_y))
                if initial_x < 110 and 180 < initial_y < 450:
                    mannequin_stationary_tracks.append(tid)
            else:
                moving_tracks.append((tid, disp, initial_x, initial_y))

        avg_moving_disp = float(np.mean([d for _, d, _, _ in moving_tracks])) if moving_tracks else 0.0
        avg_stat_disp = float(np.mean([d for _, d, _, _ in stationary_tracks])) if stationary_tracks else 0.0

        print(f"Total tracks: {total_tracks_count}")
        print(f"Moving tracks: {len(moving_tracks)}")
        print(f"Near-stationary tracks: {len(stationary_tracks)}")
        print(f"Known mannequin-region stationary tracks: {len(mannequin_stationary_tracks)}")
        print(f"Average moving-track displacement: {avg_moving_disp:.2f}")
        print(f"Average stationary-track displacement: {avg_stat_disp:.2f}")

    if save_features:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Person Tracking CSV
        tracking_csv_path = out_dir / "person_tracking.csv"
        with open(tracking_csv_path, mode="w", newline="", encoding="utf-8") as f:
            t_fields = [
                "frame", "track_id", "x1", "y1", "x2", "y2", "width", "height",
                "center_x", "center_y", "foot_x", "foot_y", "confidence",
            ]
            w = csv.DictWriter(f, fieldnames=t_fields)
            w.writeheader()
            for r in all_tracking_rows:
                w.writerow({
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

        # 2. Individual Motion Features CSVs
        person_motion_rows = []
        person_summary_rows = []
        person_motion_lookup = {}

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
                    dt = (df / streamer.fps) if df > 0 else (1.0 / streamer.fps)
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

            t_dur = (observations[-1]["frame"] - observations[0]["frame"]) / streamer.fps
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
        person_motion_csv_path = out_dir / "person_motion_features.csv"
        with open(person_motion_csv_path, mode="w", newline="", encoding="utf-8") as f:
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

        # Save Person Motion Summary CSV
        person_summary_csv_path = out_dir / "person_motion_summary.csv"
        with open(person_summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
            pms_fields = [
                "track_id", "track_duration", "observation_count", "trajectory_length",
                "total_displacement", "mean_speed", "median_speed", "speed_std",
                "max_speed", "mean_acceleration", "max_acceleration",
                "mean_direction_change", "max_direction_change",
            ]
            w = csv.DictWriter(f, fieldnames=pms_fields)
            w.writeheader()
            w.writerows(person_summary_rows)

        # 3. Person Clusters CSV
        person_clusters_csv_path = out_dir / "person_clusters.csv"
        with open(person_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
            person_fields = [
                "frame", "track_id", "foot_x", "foot_y", "cluster_id",
                "group_type", "group_id", "group_size",
                "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
            ]
            w = csv.DictWriter(f, fieldnames=person_fields)
            w.writeheader()
            for r in all_person_cluster_rows:
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

        # 4. Cluster Features CSV
        cluster_features_csv_path = out_dir / "cluster_features.csv"
        with open(cluster_features_csv_path, mode="w", newline="", encoding="utf-8") as f:
            c_fields = [
                "frame", "group_type", "group_id", "dbscan_cluster_id", "number_of_people",
                "centroid_x", "centroid_y", "min_x", "max_x", "min_y", "max_y",
                "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
                "width", "height", "area", "density",
            ]
            w = csv.DictWriter(f, fieldnames=c_fields)
            w.writeheader()
            for r in all_cluster_feature_rows:
                w.writerow({
                    "frame": r["frame"],
                    "group_type": r["group_type"],
                    "group_id": r["group_id"],
                    "dbscan_cluster_id": r["dbscan_cluster_id"],
                    "number_of_people": r["number_of_people"],
                    "centroid_x": f"{r['centroid_x']:.2f}",
                    "centroid_y": f"{r['centroid_y']:.2f}",
                    "min_x": f"{r['min_x']:.2f}",
                    "max_x": f"{r['max_x']:.2f}",
                    "min_y": f"{r['min_y']:.2f}",
                    "max_y": f"{r['max_y']:.2f}",
                    "cluster_x1": f"{r['cluster_x1']:.2f}",
                    "cluster_y1": f"{r['cluster_y1']:.2f}",
                    "cluster_x2": f"{r['cluster_x2']:.2f}",
                    "cluster_y2": f"{r['cluster_y2']:.2f}",
                    "width": f"{r['width']:.2f}",
                    "height": f"{r['height']:.2f}",
                    "area": f"{r['area']:.2f}",
                    "density": f"{r['density']:.6f}",
                })

        # 5. Persistent Clusters CSV
        persistent_clusters_csv_path = out_dir / "persistent_clusters.csv"
        with open(persistent_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
            p_fields = [
                "frame", "group_type", "group_id", "persistent_cluster_id", "dbscan_cluster_id",
                "number_of_people", "centroid_x", "centroid_y", "min_x", "max_x", "min_y", "max_y",
                "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
                "width", "height", "area", "density",
            ]
            w = csv.DictWriter(f, fieldnames=p_fields)
            w.writeheader()
            for r in all_persistent_cluster_rows:
                w.writerow({
                    "frame": r["frame"],
                    "group_type": r["group_type"],
                    "group_id": r["group_id"],
                    "persistent_cluster_id": r["persistent_cluster_id"],
                    "dbscan_cluster_id": r["dbscan_cluster_id"],
                    "number_of_people": r["number_of_people"],
                    "centroid_x": f"{r['centroid_x']:.2f}",
                    "centroid_y": f"{r['centroid_y']:.2f}",
                    "min_x": f"{r['min_x']:.2f}",
                    "max_x": f"{r['max_x']:.2f}",
                    "min_y": f"{r['min_y']:.2f}",
                    "max_y": f"{r['max_y']:.2f}",
                    "cluster_x1": f"{r['cluster_x1']:.2f}",
                    "cluster_y1": f"{r['cluster_y1']:.2f}",
                    "cluster_x2": f"{r['cluster_x2']:.2f}",
                    "cluster_y2": f"{r['cluster_y2']:.2f}",
                    "width": f"{r['width']:.2f}",
                    "height": f"{r['height']:.2f}",
                    "area": f"{r['area']:.2f}",
                    "density": f"{r['density']:.6f}",
                })

        # 6. Spatial-temporal Cluster Features
        cluster_spatiotemporal_rows = []
        cluster_summary_rows = []

        for pid, observations in sorted(persistent_cluster_obs.items()):
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
                    dt = (df / streamer.fps) if df > 0 else (1.0 / streamer.fps)
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
                            diff = abs(wrap_angle_deg(curr_d - float(prev_dir)))
                            dir_change = diff
                            dir_changes.append(diff)
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
                member_tids = cluster_member_tracks.get((cf, pid), [])
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
                    "max_individual_speed": f"{max_individual_speed:.2f}" if "max_individual_speed" in locals() else f"{max_s:.2f}",
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

            first_f = observations[0]["frame"]
            last_f = observations[-1]["frame"]
            dur_f = last_f - first_f + 1
            dur_s = dur_f / streamer.fps
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
        cluster_st_csv_path = out_dir / "cluster_spatiotemporal_features.csv"
        with open(cluster_st_csv_path, mode="w", newline="", encoding="utf-8") as f:
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

        # Save Cluster Spatiotemporal Summary CSV
        cluster_summary_csv_path = out_dir / "cluster_spatiotemporal_summary.csv"
        with open(cluster_summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
            s_fields = [
                "persistent_cluster_id", "first_frame", "last_frame", "duration_frames", "duration_seconds",
                "mean_people", "max_people", "min_people", "mean_speed", "median_speed", "speed_std",
                "max_speed", "mean_acceleration", "max_acceleration", "mean_direction_change",
                "max_direction_change", "mean_density", "density_std", "mean_directional_coherence",
                "min_directional_coherence", "max_directional_coherence", "mean_cluster_area",
                "max_cluster_area", "min_cluster_area", "mean_dispersion", "max_dispersion",
                "total_centroid_trajectory_length", "net_centroid_displacement", "trajectory_efficiency",
            ]
            w = csv.DictWriter(f, fieldnames=s_fields)
            w.writeheader()
            w.writerows(cluster_summary_rows)

        # Save Persistent Cluster Summary CSV
        persistent_summary_csv_path = out_dir / "persistent_cluster_summary.csv"
        with open(persistent_summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=s_fields)
            w.writeheader()
            w.writerows(cluster_summary_rows)

        # 7. Metadata JSON
        meta_dict = {
            "sequence_name": Path(input_path).name,
            "input_path": str(input_path),
            "source_resolution": f"{streamer.width}x{streamer.height}",
            "source_fps": streamer.fps,
            "processed_frames": processed_frames_count,
            "processing_time_s": t_elapsed,
            "processing_fps": proc_fps,
            "device": device_display,
            "gpu_name": gpu_name,
            "total_person_detections": total_tracked_persons,
            "unique_person_tracks": len(tracks_by_id),
            "persistent_clusters_count": len(persistent_cluster_obs),
            "cluster_feature_rows_count": len(cluster_spatiotemporal_rows),
        }
        with open(out_dir / "video_metadata.json", mode="w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

    summary_dict = {
        "folder": str(input_path),
        "frames": processed_frames_count,
        "processing_fps": round(proc_fps, 2),
        "device": device_display,
        "output_video": str(out_file),
        "feature_output": str(out_file.parent / "cluster_spatiotemporal_features.csv"),
    }
    return summary_dict


def main():
    parser = argparse.ArgumentParser(
        description="General-purpose reusable crowd detection, clustering, and tracking pipeline."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to video file (.mp4, .avi) or image frame directory.")
    parser.add_argument("--output", type=str, default=None, help="Destination path for annotated video (.mp4).")
    parser.add_argument("--batch_folders", nargs="+", type=str, default=None, help="Process list of sequence folders sequentially (e.g. 1 2 3 4 5 6 7 8 9 10).")
    parser.add_argument("--dataset_root", type=str, default="datasets/Frame/Frame", help="Dataset root directory containing numbered folders.")
    parser.add_argument("--output_root", type=str, default="outputs/Frame", help="Root directory for batch output folders.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights path (default: yolov8n.pt).")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cpu', or 'cuda:0' (default: auto).")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold (default: 0.15).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640).")
    parser.add_argument("--start_frame", type=int, default=1, help="1-indexed starting frame number (default: 1).")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS if input is image directory (default: 25.0).")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision on CUDA GPUs.")
    parser.add_argument("--use_sahi", action="store_true", help="Use SAHI sliced inference for person detection.")
    parser.add_argument("--slice_height", type=int, default=320, help="SAHI slice height (default: 320).")
    parser.add_argument("--slice_width", type=int, default=320, help="SAHI slice width (default: 320).")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.20, help="SAHI overlap height ratio (default: 0.20).")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.20, help="SAHI overlap width ratio (default: 0.20).")
    parser.add_argument("--highlight_stationary", action="store_true", help="Highlight near-stationary tracks (< 5px displacement) in visual output.")
    parser.add_argument("--stationary_threshold", type=float, default=5.0, help="Threshold in pixels for stationary track classification.")
    parser.add_argument("--no_features", action="store_true", help="Disable saving CSVs and spatial-temporal features.")

    args = parser.parse_args()

    save_features = not args.no_features

    if args.batch_folders:
        dataset_root = Path(args.dataset_root)
        output_root = Path(args.output_root)

        total_requested = len(args.batch_folders)
        successful_reports = []
        failed_reports = []

        print("=" * 100)
        print(f"BATCH EXECUTION PIPELINE: Processing folders {args.batch_folders}")
        print("=" * 100)

        # Pre-load model once for batch execution
        preloaded_m = None
        if not args.use_sahi:
            print(f"Pre-loading YOLO model [{args.model}] for batch execution...")
            preloaded_m = YOLO(args.model)
            print("Model pre-loaded successfully.\n")

        for fld in args.batch_folders:
            in_folder = dataset_root / fld
            out_folder = output_root / fld
            out_video = out_folder / "result.mp4"

            print(f"\n{'='*70}\n>>> STARTING FOLDER: {fld} ({in_folder} -> {out_video})\n{'='*70}")
            try:
                res = run_pipeline(
                    input_path=str(in_folder),
                    output_path=str(out_video),
                    model_path=args.model,
                    device_arg=args.device,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    start_frame=args.start_frame,
                    max_frames=args.max_frames,
                    fps_arg=args.fps,
                    use_half=args.half,
                    use_sahi=args.use_sahi,
                    slice_height=args.slice_height,
                    slice_width=args.slice_width,
                    overlap_height_ratio=args.overlap_height_ratio,
                    overlap_width_ratio=args.overlap_width_ratio,
                    highlight_stationary=args.highlight_stationary,
                    stationary_threshold=args.stationary_threshold,
                    save_features=save_features,
                    preloaded_model=preloaded_m,
                )
                res["folder_id"] = fld
                successful_reports.append(res)
                print(f">>> Successfully completed folder {fld}")
            except Exception as e:
                err_msg = str(e)
                print(f">>> ERROR: Folder {fld} failed: {err_msg}")
                failed_reports.append({
                    "folder_id": fld,
                    "folder": str(in_folder),
                    "error": err_msg,
                })
                continue

        # Final report table
        print("\n" + "=" * 100)
        print("FINAL BATCH EXECUTION REPORT")
        print("=" * 100)
        print(f"{'Folder':<10} | {'Frames':<10} | {'Processing FPS':<16} | {'Device':<10} | {'Output Video':<30} | {'Feature Output'}")
        print("-" * 100)
        for rep in successful_reports:
            fld_id = rep.get("folder_id", "")
            n_frames = rep.get("frames", 0)
            p_fps = f"{rep.get('processing_fps', 0.0):.2f}"
            dev = "CUDA" if "GPU" in rep.get("device", "") else "CPU"
            out_v = rep.get("output_video", "")
            out_f = rep.get("feature_output", "")
            print(f"{fld_id:<10} | {n_frames:<10} | {p_fps:<16} | {dev:<10} | {out_v:<30} | {out_f}")
        for fail in failed_reports:
            fld_id = fail.get("folder_id", "")
            print(f"{fld_id:<10} | {'FAILED':<10} | {'N/A':<16} | {'N/A':<10} | {'N/A':<30} | Error: {fail.get('error')}")
        print("-" * 100)
        print(f"Total folders processed: {total_requested}")
        print(f"Successful: {len(successful_reports)}")
        print(f"Failed: {len(failed_reports)}")
        print("=" * 100)
        print("Execution complete. STOPPING after folder 10 as requested.\n")

    else:
        if not args.input or not args.output:
            parser.error("Either --batch_folders or both --input and --output must be provided.")

        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            device_arg=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            fps_arg=args.fps,
            use_half=args.half,
            use_sahi=args.use_sahi,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            highlight_stationary=args.highlight_stationary,
            stationary_threshold=args.stationary_threshold,
            save_features=save_features,
        )


if __name__ == "__main__":
    main()
