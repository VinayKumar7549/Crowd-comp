import argparse
import math
from pathlib import Path
import re
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
import torch
from ultralytics import YOLO


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

    # Load YOLO model ONCE
    model = YOLO(model_path)

    # Persistent group tracking states
    active_crowd_tracks: dict[int, dict] = {}
    next_persistent_crowd_id = 0

    active_sg_tracks: dict[int, dict] = {}
    next_persistent_sg_id = 0

    processed_frames_count = 0
    total_raw_detected_persons = 0
    total_tracked_persons = 0
    total_crowd_clusters = 0
    total_small_groups = 0
    total_isolated_persons = 0
    t_start = time.time()

    active_imgsz = imgsz

    try:
        for f_idx, frame in streamer.stream():
            processed_frames_count += 1

            # Step 1: YOLO Person Detection + ByteTrack
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

            people_detections = []
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                tids = boxes.id.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()
                for tid, (x1, y1, x2, y2) in zip(tids, xyxys):
                    people_detections.append({
                        "track_id": int(tid),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "foot_x": float((x1 + x2) / 2.0),
                        "foot_y": float(y2),
                    })

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

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)
                label = f"Person {p['track_id']}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(frame, (px1, max(0, py1 - th - 4)), (px1 + tw + 4, py1), (50, 50, 50), -1)
                cv2.putText(
                    frame,
                    label,
                    (px1 + 2, py1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

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
    gpu_name = torch.cuda.get_device_name(0) if cuda_active else "None"
    duration_sec = (processed_frames_count / streamer.fps) if streamer.fps > 0 else 0.0
    end_f = start_frame + processed_frames_count - 1

    print("Input:", input_path)
    print(f"Frame range: {start_frame} to {end_f}")
    print("Frames processed:", processed_frames_count)
    print(f"FPS: {streamer.fps:.1f}")
    print(f"Duration: {duration_sec:.1f} seconds")
    print(f"Average detected persons/frame: {avg_raw_detected:.1f}")
    print("Device:", device_display)
    print("GPU:", gpu_name)
    print("Output:", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="General-purpose reusable crowd detection, clustering, and tracking pipeline."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to video file (.mp4, .avi) or image frame directory.")
    parser.add_argument("--output", type=str, required=True, help="Destination path for annotated video (.mp4).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights path (default: yolov8n.pt).")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cpu', or 'cuda:0' (default: auto).")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold (default: 0.15).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640).")
    parser.add_argument("--start_frame", type=int, default=1, help="1-indexed starting frame number (default: 1).")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS if input is image directory (default: 25.0).")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision on CUDA GPUs.")

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
