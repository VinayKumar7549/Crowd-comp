import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from detect_video import discover_avenue_videos


def get_cluster_color(cluster_id: int | str) -> tuple[int, int, int]:
    """Return distinct BGR color for clusters and groups."""
    palette = [
        (255, 100, 0),    # Vivid Blue (BGR)
        (0, 200, 50),     # Green
        (0, 140, 255),    # Orange
        (200, 0, 200),    # Magenta
        (0, 230, 255),    # Yellow
        (255, 0, 100),    # Cyan/Purple
        (50, 100, 255),   # Red-Orange
        (180, 220, 0),    # Teal
    ]
    if isinstance(cluster_id, str):
        # Hash string IDs consistently
        h = abs(hash(cluster_id))
        return palette[h % len(palette)]
    if cluster_id == -1:
        return (160, 160, 160)  # Neutral Gray
    return palette[int(cluster_id) % len(palette)]


def get_small_group_color(sg_id: int | str) -> tuple[int, int, int]:
    """Return distinct BGR color for small groups (warm amber/cyan accents)."""
    sg_palette = [
        (0, 180, 255),    # Warm Amber / Orange-Yellow
        (255, 200, 0),    # Electric Cyan
        (180, 100, 255),  # Soft Pink / Violet
        (50, 220, 150),   # Mint Green
        (200, 160, 0),    # Bright Sky Blue
    ]
    if isinstance(sg_id, str):
        h = abs(hash(sg_id))
        return sg_palette[h % len(sg_palette)]
    return sg_palette[int(sg_id) % len(sg_palette)]


def run_dbscan_comparison(
    frame_people: dict[int, list[dict]],
    eps_values: list[float],
    min_samples: int = 3,
    small_group_eps: float = 60.0,
) -> list[dict]:
    """Run DBSCAN spatial clustering over multiple eps values for parameter sensitivity analysis."""
    comparison_results = []
    total_people_all_frames = sum(len(p) for p in frame_people.values())

    for eps in eps_values:
        total_clusters = 0
        clusters_per_frame = []
        cluster_sizes = []
        small_group_count = 0
        isolated_count = 0

        for frame_idx, people in sorted(frame_people.items()):
            if len(people) < min_samples:
                labels = np.full(len(people), -1, dtype=int)
            else:
                coords = np.array([[p["foot_x"], p["foot_y"]] for p in people])
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                labels = db.labels_

            unique_clusters = [l for l in set(labels) if l != -1]
            clusters_per_frame.append(len(unique_clusters))
            total_clusters += len(unique_clusters)

            for l in unique_clusters:
                cluster_sizes.append(int((labels == l).sum()))

            # Small group disambiguation for noise
            noise_indices = [i for i, l in enumerate(labels) if l == -1]
            if len(noise_indices) >= 2:
                noise_coords = np.array([[people[i]["foot_x"], people[i]["foot_y"]] for i in noise_indices])
                sg_db = DBSCAN(eps=small_group_eps, min_samples=2).fit(noise_coords)
                sg_labels = sg_db.labels_
                small_group_count += int((sg_labels != -1).sum())
                isolated_count += int((sg_labels == -1).sum())
            else:
                isolated_count += len(noise_indices)

        avg_clusters = float(np.mean(clusters_per_frame)) if clusters_per_frame else 0.0
        avg_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        isolated_pct = (isolated_count / total_people_all_frames * 100.0) if total_people_all_frames > 0 else 0.0

        comparison_results.append({
            "eps": eps,
            "min_samples": min_samples,
            "total_clusters": total_clusters,
            "avg_cluster_count": avg_clusters,
            "avg_cluster_size": avg_size,
            "small_group_people": small_group_count,
            "isolated_people": isolated_count,
            "isolated_percentage": isolated_pct,
        })

    return comparison_results


def cluster_spatial_crowd(
    tracking_csv_path: str | Path = "outputs/tracking/avenue_tracking.csv",
    video_path: str | Path | None = None,
    output_dir: str | Path = "outputs/clustering",
    eps: float = 60.0,
    min_samples: int = 3,
    small_group_eps: float = 60.0,
    max_frames: int | None = None,
) -> dict:
    """Execute frame-level DBSCAN spatial crowd clustering, generate CSV features and annotated video.

    Distinguishes:
      - CROWD CLUSTERS (DBSCAN cluster with 3+ members)
      - SMALL GROUPS (DBSCAN noise containing 2+ nearby people within small_group_eps)
      - ISOLATED PERSONS (DBSCAN noise with 1 person)

    Args:
        tracking_csv_path: Path to tracking CSV file.
        video_path: Path to original Avenue video. If None, auto-discovers training video.
        output_dir: Output directory for CSVs, video, and plots.
        eps: DBSCAN maximum distance (in image pixels) between two foot points.
        min_samples: DBSCAN minimum number of people required to form a spatial cluster.
        small_group_eps: Distance threshold (in image pixels) to form a small group from noise points.
        max_frames: Maximum number of frames to process for video/analysis.

    Returns:
        Dictionary of summary statistics and artifact locations.
    """
    input_csv = Path(tracking_csv_path)
    if not input_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found at: {input_csv}")

    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    person_clusters_csv_path = out_dir / "person_clusters.csv"
    cluster_features_csv_path = out_dir / "cluster_features.csv"
    annotated_video_path = out_dir / "avenue_clusters.mp4"

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

    # 2. Read and group tracking data by frame
    frame_people: dict[int, list[dict]] = defaultdict(list)
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            if max_frames is not None and f_idx > max_frames:
                continue
            frame_people[f_idx].append({
                "frame": f_idx,
                "track_id": int(row["track_id"]),
                "foot_x": float(row["foot_x"]),
                "foot_y": float(row["foot_y"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            })

    total_frames_processed = len(frame_people)
    if total_frames_processed == 0:
        raise ValueError("No valid frame tracking records found.")

    # 3. Perform frame-by-frame DBSCAN clustering & small group disambiguation
    person_cluster_rows = []
    cluster_feature_rows = []

    # Diagnostic accumulators
    clusters_per_frame_list = []
    small_groups_per_frame_list = []
    isolated_per_frame_list = []
    people_in_crowd_clusters_over_time = []
    people_in_small_groups_over_time = []
    people_in_isolated_over_time = []
    all_cluster_sizes = []
    all_cluster_densities = []

    total_detected_persons = 0
    total_persons_in_crowd = 0
    total_persons_in_small_groups = 0
    total_isolated_persons = 0

    total_crowd_clusters_count = 0
    total_small_groups_count = 0

    frame_crowd_clusters_map: dict[int, list[dict]] = defaultdict(list)
    frame_small_groups_map: dict[int, list[dict]] = defaultdict(list)
    frame_isolated_map: dict[int, list[dict]] = defaultdict(list)

    for frame_idx in sorted(frame_people.keys()):
        people = frame_people[frame_idx]
        total_detected_persons += len(people)

        # DBSCAN step
        if len(people) < min_samples:
            labels = np.full(len(people), -1, dtype=int)
        else:
            coords = np.array([[p["foot_x"], p["foot_y"]] for p in people])
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = db.labels_

        # Process Crowd Clusters (DBSCAN label >= 0, 3+ members)
        unique_cids = sorted([cid for cid in set(labels) if cid != -1])
        clusters_per_frame_list.append(len(unique_cids))
        total_crowd_clusters_count += len(unique_cids)

        frame_crowd_clusters = []
        for cid in unique_cids:
            members = [p for p, l in zip(people, labels) if l == cid]
            n_people = len(members)
            foot_xs = [m["foot_x"] for m in members]
            foot_ys = [m["foot_y"] for m in members]

            # Full body bounding box covering ALL member visible bodies
            cluster_x1 = float(min(m["x1"] for m in members))
            cluster_y1 = float(min(m["y1"] for m in members))
            cluster_x2 = float(max(m["x2"] for m in members))
            cluster_y2 = float(max(m["y2"] for m in members))

            centroid_x = float(np.mean(foot_xs))
            centroid_y = float(np.mean(foot_ys))
            min_x = float(np.min(foot_xs))
            max_x = float(np.max(foot_xs))
            min_y = float(np.min(foot_ys))
            max_y = float(np.max(foot_ys))

            w = cluster_x2 - cluster_x1
            h = cluster_y2 - cluster_y1
            area = w * h
            density = (n_people / area) if area > 0 else 0.0

            all_cluster_sizes.append(n_people)
            all_cluster_densities.append(density)

            c_info = {
                "frame": frame_idx,
                "group_type": "crowd_cluster",
                "group_id": cid,
                "dbscan_cluster_id": cid,
                "number_of_people": n_people,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "cluster_x1": cluster_x1,
                "cluster_y1": cluster_y1,
                "cluster_x2": cluster_x2,
                "cluster_y2": cluster_y2,
                "width": w,
                "height": h,
                "area": area,
                "density": density,
                "members": members,
            }
            frame_crowd_clusters.append(c_info)
            cluster_feature_rows.append(c_info)

            # Assign to member dicts
            for m in members:
                m["group_type"] = "crowd_cluster"
                m["group_id"] = cid
                m["group_size"] = n_people
                m["cluster_x1"] = cluster_x1
                m["cluster_y1"] = cluster_y1
                m["cluster_x2"] = cluster_x2
                m["cluster_y2"] = cluster_y2
                m["cluster_id"] = cid

        frame_crowd_clusters_map[frame_idx] = frame_crowd_clusters
        persons_in_crowd = sum(len(c["members"]) for c in frame_crowd_clusters)
        total_persons_in_crowd += persons_in_crowd
        people_in_crowd_clusters_over_time.append(persons_in_crowd)

        # Process Noise Points -> Small Groups or Isolated Persons
        noise_indices = [i for i, l in enumerate(labels) if l == -1]
        frame_small_groups = []
        frame_isolated = []

        if len(noise_indices) >= 2:
            noise_people = [people[i] for i in noise_indices]
            noise_coords = np.array([[p["foot_x"], p["foot_y"]] for p in noise_people])
            sg_db = DBSCAN(eps=small_group_eps, min_samples=2).fit(noise_coords)
            sg_labels = sg_db.labels_

            unique_sg_ids = sorted([sg for sg in set(sg_labels) if sg != -1])
            for sg_id in unique_sg_ids:
                sg_members = [p for p, l in zip(noise_people, sg_labels) if l == sg_id]
                n_sg = len(sg_members)
                foot_xs = [m["foot_x"] for m in sg_members]
                foot_ys = [m["foot_y"] for m in sg_members]

                # Full body bounding box covering all small group members
                sg_x1 = float(min(m["x1"] for m in sg_members))
                sg_y1 = float(min(m["y1"] for m in sg_members))
                sg_x2 = float(max(m["x2"] for m in sg_members))
                sg_y2 = float(max(m["y2"] for m in sg_members))

                cx = float(np.mean(foot_xs))
                cy = float(np.mean(foot_ys))
                w = sg_x2 - sg_x1
                h = sg_y2 - sg_y1
                area = w * h
                density = (n_sg / area) if area > 0 else 0.0

                sg_info = {
                    "frame": frame_idx,
                    "group_type": "small_group",
                    "group_id": f"SG_{sg_id}",
                    "dbscan_cluster_id": -1,
                    "number_of_people": n_sg,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "min_x": float(np.min(foot_xs)),
                    "max_x": float(np.max(foot_xs)),
                    "min_y": float(np.min(foot_ys)),
                    "max_y": float(np.max(foot_ys)),
                    "cluster_x1": sg_x1,
                    "cluster_y1": sg_y1,
                    "cluster_x2": sg_x2,
                    "cluster_y2": sg_y2,
                    "width": w,
                    "height": h,
                    "area": area,
                    "density": density,
                    "members": sg_members,
                }
                frame_small_groups.append(sg_info)
                cluster_feature_rows.append(sg_info)

                for m in sg_members:
                    m["group_type"] = "small_group"
                    m["group_id"] = f"SG_{sg_id}"
                    m["group_size"] = n_sg
                    m["cluster_x1"] = sg_x1
                    m["cluster_y1"] = sg_y1
                    m["cluster_x2"] = sg_x2
                    m["cluster_y2"] = sg_y2
                    m["cluster_id"] = -1

            # Isolated persons among noise (sg_label == -1)
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
                    frame_isolated.append(p)

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
            frame_isolated.append(p)

        frame_small_groups_map[frame_idx] = frame_small_groups
        frame_isolated_map[frame_idx] = frame_isolated

        n_sg_people = sum(len(sg["members"]) for sg in frame_small_groups)
        n_iso_people = len(frame_isolated)

        total_small_groups_count += len(frame_small_groups)
        total_persons_in_small_groups += n_sg_people
        total_isolated_persons += n_iso_people

        small_groups_per_frame_list.append(len(frame_small_groups))
        isolated_per_frame_list.append(n_iso_people)
        people_in_small_groups_over_time.append(n_sg_people)
        people_in_isolated_over_time.append(n_iso_people)

        # Record person-level cluster assignments
        for person in people:
            person_cluster_rows.append({
                "frame": frame_idx,
                "track_id": person["track_id"],
                "foot_x": f"{person['foot_x']:.2f}",
                "foot_y": f"{person['foot_y']:.2f}",
                "cluster_id": person.get("cluster_id", -1),
                "group_type": person["group_type"],
                "group_id": person["group_id"],
                "group_size": person["group_size"],
                "cluster_x1": f"{person['cluster_x1']:.2f}",
                "cluster_y1": f"{person['cluster_y1']:.2f}",
                "cluster_x2": f"{person['cluster_x2']:.2f}",
                "cluster_y2": f"{person['cluster_y2']:.2f}",
            })

    # 4. Write Output 1: person_clusters.csv
    person_cols = [
        "frame", "track_id", "foot_x", "foot_y", "cluster_id",
        "group_type", "group_id", "group_size",
        "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
    ]
    with open(person_clusters_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=person_cols)
        writer.writeheader()
        writer.writerows(person_cluster_rows)

    # 5. Write Output 2: cluster_features.csv
    cluster_cols = [
        "frame", "group_type", "group_id", "dbscan_cluster_id", "number_of_people",
        "centroid_x", "centroid_y", "min_x", "max_x", "min_y", "max_y",
        "cluster_x1", "cluster_y1", "cluster_x2", "cluster_y2",
        "width", "height", "area", "density",
    ]
    with open(cluster_features_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cluster_cols, extrasaction="ignore")
        writer.writeheader()
        for r in cluster_feature_rows:
            writer.writerow({
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

    # 6. Write Output 3: Annotated Clustering Video (avenue_clusters.mp4)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_file}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (vid_w, vid_h))
    if not writer.isOpened():
        annotated_video_path = out_dir / "avenue_clusters.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (vid_w, vid_h))

    current_frame = 0
    representative_frame_img = None
    representative_frame_idx = sorted(frame_people.keys())[len(frame_people) // 2]

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            current_frame += 1

            if current_frame in frame_people:
                people = frame_people[current_frame]
                c_clusters = frame_crowd_clusters_map.get(current_frame, [])
                s_groups = frame_small_groups_map.get(current_frame, [])
                iso_persons = frame_isolated_map.get(current_frame, [])

                # 1. Draw CROWD CLUSTER bounding boxes (Requirement 1: Enclosing full bodies of all members)
                for c in c_clusters:
                    cid = c["group_id"]
                    color = get_cluster_color(cid)
                    pad = 6
                    bx1 = max(0, int(c["cluster_x1"]) - pad)
                    by1 = max(0, int(c["cluster_y1"]) - pad)
                    bx2 = min(vid_w - 1, int(c["cluster_x2"]) + pad)
                    by2 = min(vid_h - 1, int(c["cluster_y2"]) + pad)

                    # Prominent Cluster Bounding Box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                    # Centroid marker
                    cx, cy = int(c["centroid_x"]), int(c["centroid_y"])
                    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

                    # Cluster Label: Cluster <cid> | N=<count>
                    c_label = f"Cluster {cid} | N={c['number_of_people']}"
                    (tw, th), _ = cv2.getTextSize(c_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
                    cv2.putText(
                        frame,
                        c_label,
                        (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # 2. Draw SMALL GROUP bounding boxes (Requirement 2 & Small Group Handling)
                for sg in s_groups:
                    sg_id = sg["group_id"]
                    color = get_small_group_color(sg_id)
                    pad = 4
                    bx1 = max(0, int(sg["cluster_x1"]) - pad)
                    by1 = max(0, int(sg["cluster_y1"]) - pad)
                    bx2 = min(vid_w - 1, int(sg["cluster_x2"]) + pad)
                    by2 = min(vid_h - 1, int(sg["cluster_y2"]) + pad)

                    # Small Group Bounding Box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                    # Centroid marker
                    cx, cy = int(sg["centroid_x"]), int(sg["centroid_y"])
                    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_DIAMOND, 10, 1)

                    # Small Group Label: Small Group <ID> | N=<count>
                    sg_label = f"Small Group {sg_id} | N={sg['number_of_people']}"
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

                # 3. Draw ISOLATED PERSON individual bounding boxes (Requirement 2: Do NOT discard single people)
                for p in iso_persons:
                    color = (180, 180, 180)  # Subtle Slate / Silver
                    px1 = max(0, int(p["x1"]))
                    py1 = max(0, int(p["y1"]))
                    px2 = min(vid_w - 1, int(p["x2"]))
                    py2 = min(vid_h - 1, int(p["y2"]))

                    # Individual bounding box
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

                # 4. Draw foot points for debugging / research localization
                for p in people:
                    gtype = p.get("group_type", "isolated_person")
                    fx, fy = int(round(p["foot_x"])), int(round(p["foot_y"]))

                    if gtype == "crowd_cluster":
                        c_color = get_cluster_color(p.get("group_id", 0))
                        cv2.circle(frame, (fx, fy), 4, c_color, -1)
                        cv2.circle(frame, (fx, fy), 5, (255, 255, 255), 1)
                    elif gtype == "small_group":
                        sg_color = get_small_group_color(p.get("group_id", 0))
                        cv2.circle(frame, (fx, fy), 4, sg_color, -1)
                        cv2.circle(frame, (fx, fy), 5, (0, 0, 0), 1)
                    else:
                        cv2.circle(frame, (fx, fy), 3, (180, 180, 180), -1)
                        cv2.drawMarker(frame, (fx, fy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 4, 1)

                # 5. Modern HUD Panel (Top-Left)
                hud_x, hud_y = 10, 10
                hud_lines = [
                    f"Frame: {current_frame}",
                    f"Spatial DBSCAN eps={eps:.0f}px (min_samples={min_samples})",
                    f"Small-Group eps={small_group_eps:.0f}px",
                    f"Active Crowd Clusters: {len(c_clusters)}",
                    f"Active Small Groups:   {len(s_groups)}",
                    f"Isolated Persons:      {len(iso_persons)}",
                ]

                hud_h = len(hud_lines) * 18 + 12
                hud_w = 270
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

                if current_frame == representative_frame_idx:
                    representative_frame_img = frame.copy()

                writer.write(frame)

            if max_frames is not None and current_frame >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    # 7. Write Diagnostic Plots
    # Plot 1: Spatial Group Size Distribution
    plt.figure(figsize=(7, 4.5))
    if all_cluster_sizes:
        bins = np.arange(min(all_cluster_sizes) - 0.5, max(all_cluster_sizes) + 1.5, 1)
        plt.hist(all_cluster_sizes, bins=bins, color="#2563EB", edgecolor="black", alpha=0.8, rwidth=0.8)
        plt.axvline(np.mean(all_cluster_sizes), color="#DC2626", linestyle="--", label=f"Mean: {np.mean(all_cluster_sizes):.1f} people")
    plt.title(f"Crowd Cluster Size Distribution (eps={eps:.0f}, min_samples={min_samples})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of People in Cluster")
    plt.ylabel("Cluster Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_size_distribution.png", dpi=150)
    plt.close()

    # Plot 2: People Distribution Over Time (Crowd vs Small Group vs Isolated)
    plt.figure(figsize=(9, 4.5))
    frames_seq = sorted(frame_people.keys())
    plt.plot(frames_seq, people_in_crowd_clusters_over_time, color="#16A34A", linewidth=2, label="In Crowd Clusters (N>=3)")
    plt.plot(frames_seq, people_in_small_groups_over_time, color="#F59E0B", linewidth=2, label="In Small Groups (N=2)")
    plt.plot(frames_seq, people_in_isolated_over_time, color="#64748B", linestyle="--", linewidth=1.8, label="Isolated Persons (N=1)")
    plt.title("Pedestrian Population by Group Type Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index")
    plt.ylabel("Person Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "people_per_cluster_over_time.png", dpi=150)
    plt.close()

    # 8. Summary Statistics Calculation
    avg_clusters_per_frame = float(np.mean(clusters_per_frame_list)) if clusters_per_frame_list else 0.0
    max_clusters_one_frame = int(np.max(clusters_per_frame_list)) if clusters_per_frame_list else 0
    avg_cluster_size = float(np.mean(all_cluster_sizes)) if all_cluster_sizes else 0.0
    max_cluster_size = int(np.max(all_cluster_sizes)) if all_cluster_sizes else 0

    pct_crowd = (total_persons_in_crowd / total_detected_persons * 100.0) if total_detected_persons > 0 else 0.0
    pct_small_group = (total_persons_in_small_groups / total_detected_persons * 100.0) if total_detected_persons > 0 else 0.0
    pct_isolated = (total_isolated_persons / total_detected_persons * 100.0) if total_detected_persons > 0 else 0.0

    print("=" * 65)
    print("Spatial Crowd Clustering & Group Disambiguation Summary")
    print("=" * 65)
    print(f"Total frames processed:           {total_frames_processed}")
    print(f"Total detected persons:           {total_detected_persons}")
    print(f"Persons in crowd clusters:        {total_persons_in_crowd} ({pct_crowd:.1f}%)")
    print(f"Persons in small groups:          {total_persons_in_small_groups} ({pct_small_group:.1f}%)")
    print(f"Isolated persons:                 {total_isolated_persons} ({pct_isolated:.1f}%)")
    print(f"Number of crowd clusters:         {total_crowd_clusters_count}")
    print(f"Number of small groups:           {total_small_groups_count}")
    print(f"Number of isolated persons:       {total_isolated_persons}")
    print(f"Average crowd cluster size:       {avg_cluster_size:.2f} people")
    print(f"Maximum crowd cluster size:       {max_cluster_size} people")
    print(f"Person clusters CSV:              {person_clusters_csv_path}")
    print(f"Cluster features CSV:             {cluster_features_csv_path}")
    print(f"Annotated cluster video:          {annotated_video_path}")
    print("=" * 65)

    return {
        "eps": eps,
        "min_samples": min_samples,
        "small_group_eps": small_group_eps,
        "total_frames_processed": total_frames_processed,
        "total_detected_persons": total_detected_persons,
        "persons_in_crowd_clusters": total_persons_in_crowd,
        "persons_in_small_groups": total_persons_in_small_groups,
        "isolated_persons": total_isolated_persons,
        "number_of_crowd_clusters": total_crowd_clusters_count,
        "number_of_small_groups": total_small_groups_count,
        "number_of_isolated_persons": total_isolated_persons,
        "person_clusters_csv": str(person_clusters_csv_path),
        "cluster_features_csv": str(cluster_features_csv_path),
        "annotated_video": str(annotated_video_path),
        "plots_dir": str(plots_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run DBSCAN spatial crowd clustering on pedestrian foot-point tracks with small group and isolated person support."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/tracking/avenue_tracking.csv",
        help="Path to tracking CSV (default: outputs/tracking/avenue_tracking.csv).",
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
        "--eps",
        type=float,
        default=60.0,
        help="DBSCAN eps neighborhood radius in pixels (default: 60.0).",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="DBSCAN min_samples threshold (default: 3).",
    )
    parser.add_argument(
        "--small_group_eps",
        type=float,
        default=60.0,
        help="Small group distance threshold in pixels (default: 60.0).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: None).",
    )

    args = parser.parse_args()

    cluster_spatial_crowd(
        tracking_csv_path=args.csv,
        video_path=args.video,
        output_dir=args.output_dir,
        eps=args.eps,
        min_samples=args.min_samples,
        small_group_eps=args.small_group_eps,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
