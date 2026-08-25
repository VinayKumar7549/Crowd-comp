import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from detect_video import discover_avenue_videos


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


def track_persistent_clusters(
    cluster_features_csv_path: str | Path = "outputs/clustering/cluster_features.csv",
    person_clusters_csv_path: str | Path = "outputs/clustering/person_clusters.csv",
    video_path: str | Path | None = None,
    output_dir: str | Path = "outputs/clustering",
    fps: float = 25.0,
    max_centroid_dist: float = 80.0,
    min_member_overlap: float = 0.20,
    max_gap_frames: int = 5,
    max_frames: int | None = None,
) -> dict:
    """Associate frame-local DBSCAN clusters and small groups across frames to produce persistent group IDs.

    Also ensures isolated persons are fully preserved and visualized with individual bounding boxes.

    Args:
        cluster_features_csv_path: Path to frame-level cluster features CSV.
        person_clusters_csv_path: Path to person-level cluster assignments CSV.
        video_path: Path to original Avenue video. If None, auto-discovers training video.
        output_dir: Directory where outputs and plots will be saved.
        fps: Video frames per second for cluster velocity computation.
        max_centroid_dist: Maximum allowable spatial centroid distance (in pixels) for matching.
        min_member_overlap: Jaccard overlap threshold to relax spatial gating.
        max_gap_frames: Maximum allowable frame gap before a lost cluster track is terminated.
        max_frames: Maximum frames to process for video/plots.

    Returns:
        Dictionary containing tracking metrics and output file locations.
    """
    cluster_csv = Path(cluster_features_csv_path)
    person_csv = Path(person_clusters_csv_path)
    if not cluster_csv.exists():
        raise FileNotFoundError(f"Cluster features CSV not found: {cluster_csv}")
    if not person_csv.exists():
        raise FileNotFoundError(f"Person clusters CSV not found: {person_csv}")

    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    persistent_clusters_csv_path = out_dir / "persistent_clusters.csv"
    persistent_summary_csv_path = out_dir / "persistent_cluster_summary.csv"
    annotated_video_path = out_dir / "avenue_persistent_clusters.mp4"
    group_localization_video_path = out_dir / "avenue_group_localization.mp4"

    # 1. Resolve source video
    if video_path is None:
        discovered = discover_avenue_videos()
        if not discovered:
            raise FileNotFoundError("No Avenue video found in default training directory.")
        video_file = discovered[0]
    else:
        video_file = Path(video_path)
        if not video_file.exists():
            fallback = Path("datasets/Avenue_Dataset/Avenue Dataset/training_videos") / video_file.name
            if fallback.exists():
                video_file = fallback
            else:
                raise FileNotFoundError(f"Video file not found: {video_path}")

    # 2. Read person-level records
    frame_people_map: dict[int, list[dict]] = defaultdict(list)
    cluster_members: dict[tuple[int, str], set[int]] = defaultdict(set)

    with open(person_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            if max_frames is not None and f_idx > max_frames:
                continue
            tid = int(row["track_id"])
            gtype = row.get("group_type", "crowd_cluster" if int(row.get("cluster_id", -1)) != -1 else "isolated_person")
            gid = row.get("group_id", str(row.get("cluster_id", -1)))
            cid = int(row.get("cluster_id", -1))
            fx = float(row["foot_x"])
            fy = float(row["foot_y"])

            p_entry = {
                "track_id": tid,
                "group_type": gtype,
                "group_id": gid,
                "cluster_id": cid,
                "foot_x": fx,
                "foot_y": fy,
                "cluster_x1": float(row.get("cluster_x1", fx)),
                "cluster_y1": float(row.get("cluster_y1", fy)),
                "cluster_x2": float(row.get("cluster_x2", fx)),
                "cluster_y2": float(row.get("cluster_y2", fy)),
            }
            frame_people_map[f_idx].append(p_entry)
            if gtype in ["crowd_cluster", "small_group"]:
                cluster_members[(f_idx, str(gid))].add(tid)

    # 3. Read frame-level cluster features
    crowd_clusters_by_frame: dict[int, list[dict]] = defaultdict(list)
    small_groups_by_frame: dict[int, list[dict]] = defaultdict(list)

    with open(cluster_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            if max_frames is not None and f_idx > max_frames:
                continue
            gtype = row.get("group_type", "crowd_cluster")
            gid = str(row.get("group_id", row.get("cluster_id", "0")))
            cid = int(row.get("dbscan_cluster_id", row.get("cluster_id", 0)))
            members = cluster_members.get((f_idx, gid), set())

            c_info = {
                "frame": f_idx,
                "group_type": gtype,
                "group_id": gid,
                "dbscan_cluster_id": cid,
                "number_of_people": int(row["number_of_people"]),
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "min_x": float(row.get("min_x", row.get("centroid_x"))),
                "max_x": float(row.get("max_x", row.get("centroid_x"))),
                "min_y": float(row.get("min_y", row.get("centroid_y"))),
                "max_y": float(row.get("max_y", row.get("centroid_y"))),
                "cluster_x1": float(row.get("cluster_x1", row.get("min_x", 0.0))),
                "cluster_y1": float(row.get("cluster_y1", row.get("min_y", 0.0))),
                "cluster_x2": float(row.get("cluster_x2", row.get("max_x", 0.0))),
                "cluster_y2": float(row.get("cluster_y2", row.get("max_y", 0.0))),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "area": float(row["area"]),
                "density": float(row["density"]),
                "members": members,
            }
            if gtype == "crowd_cluster":
                crowd_clusters_by_frame[f_idx].append(c_info)
            elif gtype == "small_group":
                small_groups_by_frame[f_idx].append(c_info)

    # 4. Temporal Association Loop for CROWD CLUSTERS
    active_tracks: dict[int, dict] = {}
    next_persistent_id = 0

    persistent_cluster_records = []
    track_history: dict[int, list[dict]] = defaultdict(list)

    association_events = 0
    new_clusters_created = 0
    temporarily_lost_events = 0

    all_frames = sorted(set(frame_people_map.keys()) | set(crowd_clusters_by_frame.keys()) | set(small_groups_by_frame.keys()))

    for f_idx in all_frames:
        curr_clusters = crowd_clusters_by_frame.get(f_idx, [])

        if not active_tracks and curr_clusters:
            for c in curr_clusters:
                pid = next_persistent_id
                next_persistent_id += 1
                new_clusters_created += 1

                active_tracks[pid] = {
                    "persistent_id": pid,
                    "last_frame": f_idx,
                    "centroid_x": c["centroid_x"],
                    "centroid_y": c["centroid_y"],
                    "members": c["members"],
                    "missed_frames": 0,
                }
                c["persistent_cluster_id"] = pid
                persistent_cluster_records.append({**c, "persistent_cluster_id": pid})
                track_history[pid].append({**c, "persistent_cluster_id": pid})

        elif not curr_clusters:
            for pid, trk in list(active_tracks.items()):
                trk["missed_frames"] += 1
                if trk["missed_frames"] == 1:
                    temporarily_lost_events += 1
                if trk["missed_frames"] > max_gap_frames:
                    del active_tracks[pid]

        else:
            active_pids = list(active_tracks.keys())
            cost_matrix = np.full((len(active_pids), len(curr_clusters)), 1e6, dtype=float)

            for i, pid in enumerate(active_pids):
                trk = active_tracks[pid]
                for j, c in enumerate(curr_clusters):
                    dx = c["centroid_x"] - trk["centroid_x"]
                    dy = c["centroid_y"] - trk["centroid_y"]
                    dist = math.sqrt(dx * dx + dy * dy)

                    union_members = trk["members"] | c["members"]
                    overlap = (len(trk["members"] & c["members"]) / len(union_members)) if union_members else 0.0
                    cost = dist * (1.0 - 0.5 * overlap)

                    if dist <= max_centroid_dist or (overlap >= min_member_overlap and dist <= max_centroid_dist * 1.5):
                        cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_active = set()
            matched_curr = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e5:
                    pid = active_pids[r]
                    curr_c = curr_clusters[c]
                    curr_c["persistent_cluster_id"] = pid

                    active_tracks[pid]["last_frame"] = f_idx
                    active_tracks[pid]["centroid_x"] = curr_c["centroid_x"]
                    active_tracks[pid]["centroid_y"] = curr_c["centroid_y"]
                    active_tracks[pid]["members"] = curr_c["members"]
                    active_tracks[pid]["missed_frames"] = 0

                    matched_active.add(pid)
                    matched_curr.add(c)
                    association_events += 1

                    persistent_cluster_records.append({**curr_c, "persistent_cluster_id": pid})
                    track_history[pid].append({**curr_c, "persistent_cluster_id": pid})

            for pid in active_pids:
                if pid not in matched_active:
                    active_tracks[pid]["missed_frames"] += 1
                    if active_tracks[pid]["missed_frames"] == 1:
                        temporarily_lost_events += 1
                    if active_tracks[pid]["missed_frames"] > max_gap_frames:
                        del active_tracks[pid]

            for j, c in enumerate(curr_clusters):
                if j not in matched_curr:
                    pid = next_persistent_id
                    next_persistent_id += 1
                    new_clusters_created += 1

                    active_tracks[pid] = {
                        "persistent_id": pid,
                        "last_frame": f_idx,
                        "centroid_x": c["centroid_x"],
                        "centroid_y": c["centroid_y"],
                        "members": c["members"],
                        "missed_frames": 0,
                    }
                    c["persistent_cluster_id"] = pid
                    persistent_cluster_records.append({**c, "persistent_cluster_id": pid})
                    track_history[pid].append({**c, "persistent_cluster_id": pid})

    # 5. Temporal Association for SMALL GROUPS
    active_sg_tracks: dict[int, dict] = {}
    next_sg_id = 0
    persistent_sg_records = []

    for f_idx in all_frames:
        curr_sg_list = small_groups_by_frame.get(f_idx, [])
        if not active_sg_tracks and curr_sg_list:
            for sg in curr_sg_list:
                psg_id = next_sg_id
                next_sg_id += 1
                active_sg_tracks[psg_id] = {
                    "persistent_sg_id": psg_id,
                    "last_frame": f_idx,
                    "centroid_x": sg["centroid_x"],
                    "centroid_y": sg["centroid_y"],
                    "members": sg["members"],
                    "missed_frames": 0,
                }
                sg["persistent_small_group_id"] = psg_id
                persistent_sg_records.append({**sg, "persistent_small_group_id": psg_id})
        elif not curr_sg_list:
            for psg_id, trk in list(active_sg_tracks.items()):
                trk["missed_frames"] += 1
                if trk["missed_frames"] > max_gap_frames:
                    del active_sg_tracks[psg_id]
        else:
            active_sg_ids = list(active_sg_tracks.keys())
            cost_mat = np.full((len(active_sg_ids), len(curr_sg_list)), 1e6, dtype=float)
            for i, psg_id in enumerate(active_sg_ids):
                trk = active_sg_tracks[psg_id]
                for j, sg in enumerate(curr_sg_list):
                    dist = math.sqrt((sg["centroid_x"] - trk["centroid_x"]) ** 2 + (sg["centroid_y"] - trk["centroid_y"]) ** 2)
                    union_m = trk["members"] | sg["members"]
                    overlap = (len(trk["members"] & sg["members"]) / len(union_m)) if union_m else 0.0
                    cost = dist * (1.0 - 0.5 * overlap)
                    if dist <= max_centroid_dist or (overlap >= 0.30 and dist <= max_centroid_dist * 1.5):
                        cost_mat[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_act_sg = set()
            matched_cur_sg = set()

            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e5:
                    psg_id = active_sg_ids[r]
                    curr_sg = curr_sg_list[c]
                    curr_sg["persistent_small_group_id"] = psg_id
                    active_sg_tracks[psg_id]["last_frame"] = f_idx
                    active_sg_tracks[psg_id]["centroid_x"] = curr_sg["centroid_x"]
                    active_sg_tracks[psg_id]["centroid_y"] = curr_sg["centroid_y"]
                    active_sg_tracks[psg_id]["members"] = curr_sg["members"]
                    active_sg_tracks[psg_id]["missed_frames"] = 0
                    matched_act_sg.add(psg_id)
                    matched_cur_sg.add(c)
                    persistent_sg_records.append({**curr_sg, "persistent_small_group_id": psg_id})

            for psg_id in active_sg_ids:
                if psg_id not in matched_act_sg:
                    active_sg_tracks[psg_id]["missed_frames"] += 1
                    if active_sg_tracks[psg_id]["missed_frames"] > max_gap_frames:
                        del active_sg_tracks[psg_id]

            for j, sg in enumerate(curr_sg_list):
                if j not in matched_cur_sg:
                    psg_id = next_sg_id
                    next_sg_id += 1
                    active_sg_tracks[psg_id] = {
                        "persistent_sg_id": psg_id,
                        "last_frame": f_idx,
                        "centroid_x": sg["centroid_x"],
                        "centroid_y": sg["centroid_y"],
                        "members": sg["members"],
                        "missed_frames": 0,
                    }
                    sg["persistent_small_group_id"] = psg_id
                    persistent_sg_records.append({**sg, "persistent_small_group_id": psg_id})

    # 6. Write Output 1: persistent_clusters.csv
    pers_cols = [
        "frame",
        "group_type",
        "group_id",
        "persistent_cluster_id",
        "dbscan_cluster_id",
        "number_of_people",
        "centroid_x",
        "centroid_y",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "cluster_x1",
        "cluster_y1",
        "cluster_x2",
        "cluster_y2",
        "width",
        "height",
        "area",
        "density",
    ]
    with open(persistent_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pers_cols, extrasaction="ignore")
        writer.writeheader()
        for r in persistent_cluster_records:
            writer.writerow({
                "frame": r["frame"],
                "group_type": "crowd_cluster",
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
        for sg in persistent_sg_records:
            writer.writerow({
                "frame": sg["frame"],
                "group_type": "small_group",
                "group_id": sg["group_id"],
                "persistent_cluster_id": f"SG_{sg.get('persistent_small_group_id', 0)}",
                "dbscan_cluster_id": -1,
                "number_of_people": sg["number_of_people"],
                "centroid_x": f"{sg['centroid_x']:.2f}",
                "centroid_y": f"{sg['centroid_y']:.2f}",
                "min_x": f"{sg['min_x']:.2f}",
                "max_x": f"{sg['max_x']:.2f}",
                "min_y": f"{sg['min_y']:.2f}",
                "max_y": f"{sg['max_y']:.2f}",
                "cluster_x1": f"{sg['cluster_x1']:.2f}",
                "cluster_y1": f"{sg['cluster_y1']:.2f}",
                "cluster_x2": f"{sg['cluster_x2']:.2f}",
                "cluster_y2": f"{sg['cluster_y2']:.2f}",
                "width": f"{sg['width']:.2f}",
                "height": f"{sg['height']:.2f}",
                "area": f"{sg['area']:.2f}",
                "density": f"{sg['density']:.6f}",
            })

    # 7. Write Output 2: persistent_cluster_summary.csv
    summary_rows = []
    for pid, obs_list in sorted(track_history.items()):
        obs_list.sort(key=lambda x: x["frame"])
        first_f = obs_list[0]["frame"]
        last_f = obs_list[-1]["frame"]
        duration_frames = last_f - first_f + 1
        num_obs_frames = len(obs_list)

        people_counts = [o["number_of_people"] for o in obs_list]
        areas = [o["area"] for o in obs_list]
        densities = [o["density"] for o in obs_list]
        cxs = [o["centroid_x"] for o in obs_list]
        cys = [o["centroid_y"] for o in obs_list]

        net_disp = math.sqrt((cxs[-1] - cxs[0]) ** 2 + (cys[-1] - cys[0]) ** 2)

        speeds = []
        for k in range(1, len(obs_list)):
            df = obs_list[k]["frame"] - obs_list[k - 1]["frame"]
            if df > 0:
                dt = df / fps
                step_d = math.sqrt((cxs[k] - cxs[k - 1]) ** 2 + (cys[k] - cys[k - 1]) ** 2)
                speeds.append(step_d / dt)

        mean_spd = float(np.mean(speeds)) if speeds else 0.0
        max_spd = float(np.max(speeds)) if speeds else 0.0

        summary_rows.append({
            "persistent_cluster_id": pid,
            "first_frame": first_f,
            "last_frame": last_f,
            "duration_frames": duration_frames,
            "number_of_frames": num_obs_frames,
            "mean_people": f"{np.mean(people_counts):.2f}",
            "max_people": int(np.max(people_counts)),
            "mean_centroid_x": f"{np.mean(cxs):.2f}",
            "mean_centroid_y": f"{np.mean(cys):.2f}",
            "centroid_displacement": f"{net_disp:.2f}",
            "mean_cluster_speed": f"{mean_spd:.2f}",
            "max_cluster_speed": f"{max_spd:.2f}",
            "mean_cluster_area": f"{np.mean(areas):.2f}",
            "mean_cluster_density": f"{np.mean(densities):.6f}",
        })

    sum_cols = [
        "persistent_cluster_id",
        "first_frame",
        "last_frame",
        "duration_frames",
        "number_of_frames",
        "mean_people",
        "max_people",
        "mean_centroid_x",
        "mean_centroid_y",
        "centroid_displacement",
        "mean_cluster_speed",
        "max_cluster_speed",
        "mean_cluster_area",
        "mean_cluster_density",
    ]
    with open(persistent_summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sum_cols)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 8. Write Output 3: Annotated Video (avenue_persistent_clusters.mp4)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_file}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if cap_fps <= 0 or cap_fps != cap_fps:
        cap_fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, cap_fps, (vid_w, vid_h))
    if not writer.isOpened():
        annotated_video_path = out_dir / "avenue_persistent_clusters.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, cap_fps, (vid_w, vid_h))

    # Fast lookup mapping: frame -> list of persistent clusters & small groups
    frame_pers_map: dict[int, list[dict]] = defaultdict(list)
    for r in persistent_cluster_records:
        frame_pers_map[r["frame"]].append(r)

    frame_pers_sg_map: dict[int, list[dict]] = defaultdict(list)
    for sg in persistent_sg_records:
        frame_pers_sg_map[sg["frame"]].append(sg)

    current_frame = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            current_frame += 1

            if current_frame in frame_people_map:
                people = frame_people_map[current_frame]
                p_clusters = frame_pers_map.get(current_frame, [])
                p_sgs = frame_pers_sg_map.get(current_frame, [])
                iso_persons = [p for p in people if p.get("group_type") == "isolated_person"]

                # 1. Draw CROWD CLUSTER bounding boxes (Requirement 1: Single box enclosing ALL members)
                for c in p_clusters:
                    pid = c["persistent_cluster_id"]
                    color = get_persistent_color(pid)
                    pad = 6
                    bx1 = max(0, int(c["cluster_x1"]) - pad)
                    by1 = max(0, int(c["cluster_y1"]) - pad)
                    bx2 = min(vid_w - 1, int(c["cluster_x2"]) + pad)
                    by2 = min(vid_h - 1, int(c["cluster_y2"]) + pad)

                    # Prominent Cluster Bounding Box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cx, cy = int(c["centroid_x"]), int(c["centroid_y"])
                    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

                    # Header Label: Group <persistent_cluster_id> | People: <N> (or Cluster <pid> | N=<count>)
                    header_label = f"Group {pid} | People: {c['number_of_people']}"
                    (tw, th), _ = cv2.getTextSize(header_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
                    cv2.putText(
                        frame,
                        header_label,
                        (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # 2. Draw SMALL GROUP bounding boxes (Requirement 2: Single box covering both/all people's full bboxes)
                for sg in p_sgs:
                    psg_id = sg.get("persistent_small_group_id", 0)
                    color = get_small_group_color(psg_id)
                    pad = 4
                    bx1 = max(0, int(sg["cluster_x1"]) - pad)
                    by1 = max(0, int(sg["cluster_y1"]) - pad)
                    bx2 = min(vid_w - 1, int(sg["cluster_x2"]) + pad)
                    by2 = min(vid_h - 1, int(sg["cluster_y2"]) + pad)

                    # Small Group Bounding Box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cx, cy = int(sg["centroid_x"]), int(sg["centroid_y"])
                    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_DIAMOND, 10, 1)

                    # Header Label: Small Group <ID> | N=2
                    sg_label = f"Small Group {psg_id} | N={sg['number_of_people']}"
                    (tw, th), _ = cv2.getTextSize(sg_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
                    cv2.putText(
                        frame,
                        sg_label,
                        (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )

                # 3. Draw ISOLATED PERSON bounding boxes (Requirement 2: Person <track_id>)
                for p in iso_persons:
                    color = (180, 180, 180)  # Subtle Silver / Slate Gray
                    px1 = max(0, int(p["cluster_x1"]))
                    py1 = max(0, int(p["cluster_y1"]))
                    px2 = min(vid_w - 1, int(p["cluster_x2"]))
                    py2 = min(vid_h - 1, int(p["cluster_y2"]))

                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

                    # Label: Person <track_id>
                    p_label = f"Person {p['track_id']}"
                    (tw, th), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(frame, (px1, max(0, py1 - th - 4)), (px1 + tw + 4, py1), (50, 50, 50), -1)
                    cv2.putText(
                        frame,
                        p_label,
                        (px1 + 2, py1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )

                # 4. Draw foot points
                for p in people:
                    gtype = p.get("group_type", "isolated_person")
                    fx, fy = int(round(p["foot_x"])), int(round(p["foot_y"]))

                    if gtype == "crowd_cluster":
                        # Find pid
                        c_match = [c for c in p_clusters if p["track_id"] in c["members"]]
                        c_color = get_persistent_color(c_match[0]["persistent_cluster_id"]) if c_match else (255, 120, 0)
                        cv2.circle(frame, (fx, fy), 4, c_color, -1)
                        cv2.circle(frame, (fx, fy), 5, (255, 255, 255), 1)
                    elif gtype == "small_group":
                        sg_match = [sg for sg in p_sgs if p["track_id"] in sg["members"]]
                        sg_color = get_small_group_color(sg_match[0].get("persistent_small_group_id", 0)) if sg_match else (0, 180, 255)
                        cv2.circle(frame, (fx, fy), 4, sg_color, -1)
                        cv2.circle(frame, (fx, fy), 5, (0, 0, 0), 1)
                    else:
                        cv2.circle(frame, (fx, fy), 3, (180, 180, 180), -1)
                        cv2.drawMarker(frame, (fx, fy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 4, 1)

                # 5. HUD Overlay Panel
                hud_x, hud_y = 10, 10
                hud_lines = [
                    f"Frame: {current_frame}",
                    f"Active Persistent Crowd Groups: {len(p_clusters)}",
                    f"Active Persistent Small Groups: {len(p_sgs)}",
                    f"Isolated Persons:               {len(iso_persons)}",
                ]

                hud_h = len(hud_lines) * 18 + 12
                hud_w = 280
                overlay = frame.copy()
                cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (90, 90, 90), 1)

                for idx, line in enumerate(hud_lines):
                    cv2.putText(
                        frame,
                        line,
                        (hud_x + 8, hud_y + 16 + idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (240, 240, 240),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(frame)

            if max_frames is not None and current_frame >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    # Create alias/copy of video as avenue_group_localization.mp4 if needed
    import shutil
    try:
        shutil.copyfile(annotated_video_path, group_localization_video_path)
    except Exception:
        pass

    # 9. Diagnostic Plots
    plt.figure(figsize=(9, 4.5))
    frame_counts = [len(frame_pers_map.get(f, [])) for f in all_frames]
    plt.plot(all_frames, frame_counts, color="#4F46E5", linewidth=2, label="Persistent Groups Count")
    plt.title("Persistent Crowd Groups Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index")
    plt.ylabel("Active Persistent Groups")
    plt.ylim(-0.2, max(frame_counts, default=1) + 1.2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "persistent_cluster_count_over_time.png", dpi=150)
    plt.close()

    # 10. Summary Statistics
    total_persistent_clusters = next_persistent_id
    durations = [int(r["duration_frames"]) for r in summary_rows]
    avg_duration = float(np.mean(durations)) if durations else 0.0
    max_duration = int(np.max(durations)) if durations else 0

    all_people_counts = [int(r["max_people"]) for r in summary_rows]
    mean_people_all = [float(r["mean_people"]) for r in summary_rows]
    avg_people_per_cluster = float(np.mean(mean_people_all)) if mean_people_all else 0.0
    max_people_per_cluster = int(np.max(all_people_counts)) if all_people_counts else 0

    print("=" * 65)
    print("Persistent Crowd Group Tracking Summary")
    print("=" * 65)
    print(f"Total persistent crowd clusters created: {total_persistent_clusters}")
    print(f"Total persistent small groups created:   {next_sg_id}")
    print(f"Average persistent cluster duration:     {avg_duration:.2f} frames")
    print(f"Maximum persistent cluster duration:     {max_duration} frames")
    print(f"Average people per cluster:             {avg_people_per_cluster:.2f} people")
    print(f"Maximum people per cluster:             {max_people_per_cluster} people")
    print(f"Number of cluster association events:   {association_events}")
    print(f"Number of new clusters created:         {new_clusters_created}")
    print(f"Number of temporarily lost clusters:    {temporarily_lost_events}")
    print(f"Persistent clusters CSV:                {persistent_clusters_csv_path}")
    print(f"Persistent summary CSV:                 {persistent_summary_csv_path}")
    print(f"Annotated video:                        {annotated_video_path}")
    print(f"Group localization video:               {group_localization_video_path}")
    print("=" * 65)

    return {
        "total_persistent_clusters": total_persistent_clusters,
        "total_persistent_small_groups": next_sg_id,
        "avg_duration": avg_duration,
        "max_duration": max_duration,
        "avg_people": avg_people_per_cluster,
        "max_people": max_people_per_cluster,
        "association_events": association_events,
        "new_clusters_created": new_clusters_created,
        "temporarily_lost_events": temporarily_lost_events,
        "persistent_clusters_csv": str(persistent_clusters_csv_path),
        "persistent_summary_csv": str(persistent_summary_csv_path),
        "annotated_video": str(annotated_video_path),
        "group_localization_video": str(group_localization_video_path),
        "plots_dir": str(plots_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Track spatial crowd clusters across frames to create persistent group IDs."
    )
    parser.add_argument(
        "--clusters_csv",
        type=str,
        default="outputs/clustering/cluster_features.csv",
        help="Path to cluster_features.csv (default: outputs/clustering/cluster_features.csv).",
    )
    parser.add_argument(
        "--persons_csv",
        type=str,
        default="outputs/clustering/person_clusters.csv",
        help="Path to person_clusters.csv (default: outputs/clustering/person_clusters.csv).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to original video file (default: first discovered training video).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/clustering",
        help="Output directory (default: outputs/clustering).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video FPS for cluster velocity calculation (default: 25.0).",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=80.0,
        help="Maximum spatial centroid distance in pixels for matching (default: 80.0).",
    )
    parser.add_argument(
        "--min_overlap",
        type=float,
        default=0.20,
        help="Minimum member Jaccard overlap threshold (default: 0.20).",
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=5,
        help="Maximum frame gap before terminating cluster track (default: 5).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: None).",
    )

    args = parser.parse_args()

    track_persistent_clusters(
        cluster_features_csv_path=args.clusters_csv,
        person_clusters_csv_path=args.persons_csv,
        video_path=args.video,
        output_dir=args.output_dir,
        fps=args.fps,
        max_centroid_dist=args.max_dist,
        min_member_overlap=args.min_overlap,
        max_gap_frames=args.max_gap,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
