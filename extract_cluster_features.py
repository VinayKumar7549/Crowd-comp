import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def wrap_angle_deg(angle_diff: float) -> float:
    """Wrap angular difference into range [-180, 180] degrees."""
    return (angle_diff + 180.0) % 360.0 - 180.0


def extract_cluster_spatiotemporal_features(
    persistent_clusters_csv: str | Path = "outputs/clustering/persistent_clusters.csv",
    person_clusters_csv: str | Path = "outputs/clustering/person_clusters.csv",
    person_motion_csv: str | Path = "outputs/features/person_motion_features.csv",
    output_dir: str | Path = "outputs/features",
    fps: float = 25.0,
    jitter_threshold: float = 0.5,
    min_speed_coherence: float = 1.0,
) -> dict:
    """Extract cluster-level spatial-temporal discriminant motion, size, dispersion, and coherence features.

    Distinguishes crowd-level features, small-group features, and individual motion features.

    Args:
        persistent_clusters_csv: Path to persistent_clusters.csv.
        person_clusters_csv: Path to person_clusters.csv.
        person_motion_csv: Path to person_motion_features.csv.
        output_dir: Directory where features, summaries, and plots will be saved.
        fps: Video FPS for scaling velocity and acceleration derivatives.
        jitter_threshold: Minimum centroid displacement in pixels to consider directional movement.
        min_speed_coherence: Minimum individual speed in px/s to include in directional coherence.

    Returns:
        Dictionary containing summary statistics, quality checks, and generated file paths.
    """
    pers_csv_path = Path(persistent_clusters_csv)
    person_cls_path = Path(person_clusters_csv)
    person_mot_path = Path(person_motion_csv)

    if not pers_csv_path.exists():
        raise FileNotFoundError(f"File not found: {pers_csv_path}")
    if not person_cls_path.exists():
        raise FileNotFoundError(f"File not found: {person_cls_path}")
    if not person_mot_path.exists():
        raise FileNotFoundError(f"File not found: {person_mot_path}")

    out_dir = Path(output_dir)
    plots_dir = out_dir / "cluster_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame_features_path = out_dir / "cluster_spatiotemporal_features.csv"
    summary_features_path = out_dir / "cluster_spatiotemporal_summary.csv"
    metadata_json_path = out_dir / "cluster_feature_metadata.json"

    # 1. Load person motion features mapped by (frame, track_id)
    person_motion_map: dict[tuple[int, int], dict] = {}
    with open(person_mot_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            tid = int(row["track_id"])
            person_motion_map[(f_idx, tid)] = {
                "foot_x": float(row["foot_x"]),
                "foot_y": float(row["foot_y"]),
                "velocity_x": float(row["velocity_x"]),
                "velocity_y": float(row["velocity_y"]),
                "speed": float(row["speed"]),
            }

    # 2. Map (frame, group_id / cluster_id) -> list of member track IDs from person_clusters.csv
    cluster_member_tracks: dict[tuple[int, str], list[int]] = defaultdict(list)
    with open(person_cls_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            tid = int(row["track_id"])
            gid = str(row.get("group_id", row.get("cluster_id", "-1")))
            cid = str(row.get("cluster_id", "-1"))
            gtype = row.get("group_type", "")
            if gtype != "isolated_person" and gid != "-1":
                cluster_member_tracks[(f_idx, gid)].append(tid)
            if cid != "-1":
                cluster_member_tracks[(f_idx, cid)].append(tid)

    # 3. Read persistent clusters and group observations by persistent_cluster_id
    raw_clusters_by_pid: dict[str, list[dict]] = defaultdict(list)
    with open(pers_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["persistent_cluster_id"])
            f_idx = int(row["frame"])
            gtype = row.get("group_type", "crowd_cluster")
            gid = str(row.get("group_id", row.get("dbscan_cluster_id", "0")))
            dbscan_id = int(row.get("dbscan_cluster_id", -1))

            obs = {
                "frame": f_idx,
                "group_type": gtype,
                "group_id": gid,
                "dbscan_cluster_id": dbscan_id,
                "persistent_cluster_id": pid,
                "number_of_people": int(row["number_of_people"]),
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "min_x": float(row.get("min_x", 0.0)),
                "max_x": float(row.get("max_x", 0.0)),
                "min_y": float(row.get("min_y", 0.0)),
                "max_y": float(row.get("max_y", 0.0)),
                "cluster_x1": float(row.get("cluster_x1", row.get("min_x", 0.0))),
                "cluster_y1": float(row.get("cluster_y1", row.get("min_y", 0.0))),
                "cluster_x2": float(row.get("cluster_x2", row.get("max_x", 0.0))),
                "cluster_y2": float(row.get("cluster_y2", row.get("max_y", 0.0))),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "area": float(row["area"]),
                "density": float(row["density"]),
            }
            raw_clusters_by_pid[pid].append(obs)

    # 4. Calculate frame-level spatiotemporal discriminant features
    all_frame_feature_rows = []
    cluster_summary_rows = []

    total_invalid_rows = 0

    for pid, observations in sorted(raw_clusters_by_pid.items()):
        observations.sort(key=lambda x: x["frame"])

        prev_obs = None
        prev_vx = None
        prev_vy = None
        prev_direction = None

        track_speeds = []
        track_accels = []
        track_dir_changes = []
        track_densities = []
        track_coherences = []
        track_areas = []
        track_dispersions = []
        track_people = []
        track_displacements = []
        track_centroids = []

        gtype = observations[0].get("group_type", "crowd_cluster")

        for i, obs in enumerate(observations):
            curr_frame = obs["frame"]
            gid = obs["group_id"]
            dbscan_id = str(obs["dbscan_cluster_id"])
            n_people = obs["number_of_people"]
            cx = obs["centroid_x"]
            cy = obs["centroid_y"]
            cw = obs["width"]
            ch = obs["height"]
            c_area = obs["area"]
            c_density = obs["density"]

            track_centroids.append((cx, cy))
            track_people.append(n_people)
            track_areas.append(c_area)
            track_densities.append(c_density)

            # Motion Derivatives & Expansion/Contraction
            if prev_obs is None:
                cdx = 0.0
                cdy = 0.0
                c_disp = 0.0
                c_vx = 0.0
                c_vy = 0.0
                c_speed = 0.0
                c_accel = 0.0
                direction_deg_str = ""
                dir_change_deg = 0.0

                area_change = 0.0
                width_change = 0.0
                height_change = 0.0
                density_change = 0.0
            else:
                df = curr_frame - prev_obs["frame"]
                dt = (df / fps) if df > 0 else (1.0 / fps)

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
                    track_accels.append(c_accel)
                else:
                    c_accel = 0.0

                if c_disp >= jitter_threshold:
                    curr_direction = math.degrees(math.atan2(cdy, cdx))
                    direction_deg_str = f"{curr_direction:.2f}"

                    if prev_direction is not None and prev_direction != "":
                        raw_diff = curr_direction - float(prev_direction)
                        dir_change_deg = abs(wrap_angle_deg(raw_diff))
                        track_dir_changes.append(dir_change_deg)
                    else:
                        dir_change_deg = 0.0
                    prev_direction = curr_direction
                else:
                    direction_deg_str = ""
                    dir_change_deg = 0.0

                prev_vx = c_vx
                prev_vy = c_vy

                area_change = (c_area - prev_obs["area"]) / dt
                width_change = (cw - prev_obs["width"]) / dt
                height_change = (ch - prev_obs["height"]) / dt
                density_change = (c_density - prev_obs["density"]) / dt

                track_speeds.append(c_speed)
                track_displacements.append(c_disp)

            prev_obs = obs

            # Pedestrian Dispersion & Directional Coherence
            member_tids = cluster_member_tracks.get((curr_frame, gid), []) or cluster_member_tracks.get((curr_frame, dbscan_id), [])
            # Deduplicate member tids
            member_tids = list(set(member_tids))

            member_dists = []
            member_speeds = []
            member_vxs = []
            member_vys = []
            unit_vx_list = []
            unit_vy_list = []

            for tid in member_tids:
                p_mot = person_motion_map.get((curr_frame, tid))
                if p_mot:
                    px = p_mot["foot_x"]
                    py = p_mot["foot_y"]
                    dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                    member_dists.append(dist)

                    vx_i = p_mot["velocity_x"]
                    vy_i = p_mot["velocity_y"]
                    spd_i = p_mot["speed"]
                    member_speeds.append(spd_i)
                    member_vxs.append(vx_i)
                    member_vys.append(vy_i)

                    if spd_i >= min_speed_coherence:
                        unit_vx_list.append(vx_i / spd_i)
                        unit_vy_list.append(vy_i / spd_i)

            if member_dists:
                mean_dist = float(np.mean(member_dists))
                median_dist = float(np.median(member_dists))
                std_dist = float(np.std(member_dists))
                max_dist = float(np.max(member_dists))
                spatial_disp = float(np.sqrt(np.mean(np.square(member_dists))))
            else:
                mean_dist = median_dist = std_dist = max_dist = spatial_disp = 0.0

            track_dispersions.append(spatial_disp)

            if member_speeds:
                mean_ind_spd = float(np.mean(member_speeds))
                std_ind_spd = float(np.std(member_speeds))
                max_ind_spd = float(np.max(member_speeds))
                mean_vx = float(np.mean(member_vxs))
                mean_vy = float(np.mean(member_vys))
                vel_disp = float(
                    np.sqrt(
                        np.mean(
                            [(vx_i - mean_vx) ** 2 + (vy_i - mean_vy) ** 2 for vx_i, vy_i in zip(member_vxs, member_vys)]
                        )
                    )
                )
            else:
                mean_ind_spd = std_ind_spd = max_ind_spd = mean_vx = mean_vy = vel_disp = 0.0

            if unit_vx_list:
                mean_uvx = float(np.mean(unit_vx_list))
                mean_uvy = float(np.mean(unit_vy_list))
                dir_coherence = float(np.clip(math.sqrt(mean_uvx * mean_uvx + mean_uvy * mean_uvy), 0.0, 1.0))
            else:
                dir_coherence = 1.0

            track_coherences.append(dir_coherence)

            feature_row = {
                "frame": curr_frame,
                "group_type": gtype,
                "persistent_cluster_id": pid,
                "number_of_people": n_people,
                "centroid_x": f"{cx:.2f}",
                "centroid_y": f"{cy:.2f}",
                "centroid_dx": f"{cdx:.2f}",
                "centroid_dy": f"{cdy:.2f}",
                "cluster_displacement": f"{c_disp:.2f}",
                "cluster_velocity_x": f"{c_vx:.2f}",
                "cluster_velocity_y": f"{c_vy:.2f}",
                "cluster_speed": f"{c_speed:.2f}",
                "cluster_acceleration": f"{c_accel:.2f}",
                "cluster_direction": direction_deg_str,
                "cluster_direction_change": f"{dir_change_deg:.2f}",
                "cluster_width": f"{cw:.2f}",
                "cluster_height": f"{ch:.2f}",
                "cluster_area": f"{c_area:.2f}",
                "density": f"{c_density:.6f}",
                "mean_distance_to_centroid": f"{mean_dist:.2f}",
                "median_distance_to_centroid": f"{median_dist:.2f}",
                "std_distance_to_centroid": f"{std_dist:.2f}",
                "max_distance_to_centroid": f"{max_dist:.2f}",
                "spatial_dispersion": f"{spatial_disp:.2f}",
                "mean_individual_speed": f"{mean_ind_spd:.2f}",
                "std_individual_speed": f"{std_ind_spd:.2f}",
                "max_individual_speed": f"{max_ind_spd:.2f}",
                "mean_velocity_x": f"{mean_vx:.2f}",
                "mean_velocity_y": f"{mean_vy:.2f}",
                "velocity_dispersion": f"{vel_disp:.2f}",
                "directional_coherence": f"{dir_coherence:.4f}",
                "area_change": f"{area_change:.2f}",
                "width_change": f"{width_change:.2f}",
                "height_change": f"{height_change:.2f}",
                "density_change": f"{density_change:.6f}",
            }
            all_frame_feature_rows.append(feature_row)

        first_frame = observations[0]["frame"]
        last_frame = observations[-1]["frame"]
        duration_frames = last_frame - first_frame + 1
        duration_sec = duration_frames / fps

        traj_length = sum(track_displacements)
        net_disp = math.sqrt(
            (track_centroids[-1][0] - track_centroids[0][0]) ** 2
            + (track_centroids[-1][1] - track_centroids[0][1]) ** 2
        )
        traj_efficiency = (net_disp / traj_length) if traj_length > 0 else 0.0

        summary_row = {
            "persistent_cluster_id": pid,
            "group_type": gtype,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_frames": duration_frames,
            "duration_seconds": f"{duration_sec:.2f}",
            "mean_people": f"{np.mean(track_people):.2f}",
            "max_people": int(np.max(track_people)),
            "min_people": int(np.min(track_people)),
            "mean_speed": f"{np.mean(track_speeds):.2f}" if track_speeds else "0.00",
            "median_speed": f"{np.median(track_speeds):.2f}" if track_speeds else "0.00",
            "speed_std": f"{np.std(track_speeds):.2f}" if track_speeds else "0.00",
            "max_speed": f"{np.max(track_speeds):.2f}" if track_speeds else "0.00",
            "mean_acceleration": f"{np.mean(track_accels):.2f}" if track_accels else "0.00",
            "max_acceleration": f"{np.max(track_accels):.2f}" if track_accels else "0.00",
            "mean_direction_change": f"{np.mean(track_dir_changes):.2f}" if track_dir_changes else "0.00",
            "max_direction_change": f"{np.max(track_dir_changes):.2f}" if track_dir_changes else "0.00",
            "mean_density": f"{np.mean(track_densities):.6f}",
            "density_std": f"{np.std(track_densities):.6f}",
            "mean_directional_coherence": f"{np.mean(track_coherences):.4f}",
            "min_directional_coherence": f"{np.min(track_coherences):.4f}",
            "max_directional_coherence": f"{np.max(track_coherences):.4f}",
            "mean_cluster_area": f"{np.mean(track_areas):.2f}",
            "max_cluster_area": f"{np.max(track_areas):.2f}",
            "min_cluster_area": f"{np.min(track_areas):.2f}",
            "mean_dispersion": f"{np.mean(track_dispersions):.2f}",
            "max_dispersion": f"{np.max(track_dispersions):.2f}",
            "total_centroid_trajectory_length": f"{traj_length:.2f}",
            "net_centroid_displacement": f"{net_disp:.2f}",
            "trajectory_efficiency": f"{traj_efficiency:.4f}",
        }
        cluster_summary_rows.append(summary_row)

    # 5. Write Frame-Level CSV
    frame_cols = [
        "frame",
        "group_type",
        "persistent_cluster_id",
        "number_of_people",
        "centroid_x",
        "centroid_y",
        "centroid_dx",
        "centroid_dy",
        "cluster_displacement",
        "cluster_velocity_x",
        "cluster_velocity_y",
        "cluster_speed",
        "cluster_acceleration",
        "cluster_direction",
        "cluster_direction_change",
        "cluster_width",
        "cluster_height",
        "cluster_area",
        "density",
        "mean_distance_to_centroid",
        "median_distance_to_centroid",
        "std_distance_to_centroid",
        "max_distance_to_centroid",
        "spatial_dispersion",
        "mean_individual_speed",
        "std_individual_speed",
        "max_individual_speed",
        "mean_velocity_x",
        "mean_velocity_y",
        "velocity_dispersion",
        "directional_coherence",
        "area_change",
        "width_change",
        "height_change",
        "density_change",
    ]
    with open(frame_features_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_cols)
        writer.writeheader()
        writer.writerows(all_frame_feature_rows)

    # 6. Write Cluster Summary CSV
    sum_cols = [
        "persistent_cluster_id",
        "group_type",
        "first_frame",
        "last_frame",
        "duration_frames",
        "duration_seconds",
        "mean_people",
        "max_people",
        "min_people",
        "mean_speed",
        "median_speed",
        "speed_std",
        "max_speed",
        "mean_acceleration",
        "max_acceleration",
        "mean_direction_change",
        "max_direction_change",
        "mean_density",
        "density_std",
        "mean_directional_coherence",
        "min_directional_coherence",
        "max_directional_coherence",
        "mean_cluster_area",
        "max_cluster_area",
        "min_cluster_area",
        "mean_dispersion",
        "max_dispersion",
        "total_centroid_trajectory_length",
        "net_centroid_displacement",
        "trajectory_efficiency",
    ]
    with open(summary_features_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sum_cols)
        writer.writeheader()
        writer.writerows(cluster_summary_rows)

    # 7. Generate Visualizations in outputs/features/cluster_plots/
    rows_by_pid = defaultdict(list)
    for r in all_frame_feature_rows:
        rows_by_pid[r["persistent_cluster_id"]].append(r)

    palette = ["#2563EB", "#16A34A", "#EA580C", "#9333EA", "#0891B2", "#DC2626"]

    # Plot 1: Cluster Speed Over Time
    plt.figure(figsize=(9, 4.5))
    for idx, (pid, r_list) in enumerate(sorted(rows_by_pid.items())):
        fs = [r["frame"] for r in r_list]
        spds = [float(r["cluster_speed"]) for r in r_list]
        c = palette[idx % len(palette)]
        plt.plot(fs, spds, "-o", markersize=3, linewidth=1.8, label=f"Group {pid}", color=c)
    plt.title("Persistent Group Speed Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index")
    plt.ylabel("Speed (pixels/second)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_speed_over_time.png", dpi=150)
    plt.close()

    # Plot 2: Directional Coherence Over Time
    plt.figure(figsize=(9, 4.5))
    for idx, (pid, r_list) in enumerate(sorted(rows_by_pid.items())):
        fs = [r["frame"] for r in r_list]
        cohs = [float(r["directional_coherence"]) for r in r_list]
        c = palette[idx % len(palette)]
        plt.plot(fs, cohs, "-^", markersize=3, linewidth=1.8, label=f"Group {pid}", color=c)
    plt.title("Directional Motion Coherence Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index")
    plt.ylabel("Directional Coherence [0.0 - 1.0]")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_coherence_over_time.png", dpi=150)
    plt.close()

    meta = {
        "total_group_feature_rows": len(all_frame_feature_rows),
        "total_persistent_groups_summarized": len(cluster_summary_rows),
        "frame_features_csv": str(frame_features_path),
        "summary_features_csv": str(summary_features_path),
        "plots_dir": str(plots_dir),
    }
    with open(metadata_json_path, mode="w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Extract spatiotemporal features from persistent crowd clusters and small groups."
    )
    parser.add_argument(
        "--persistent_clusters",
        type=str,
        default="outputs/clustering/persistent_clusters.csv",
        help="Path to persistent_clusters.csv.",
    )
    parser.add_argument(
        "--person_clusters",
        type=str,
        default="outputs/clustering/person_clusters.csv",
        help="Path to person_clusters.csv.",
    )
    parser.add_argument(
        "--person_motion",
        type=str,
        default="outputs/features/person_motion_features.csv",
        help="Path to person_motion_features.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/features",
        help="Output directory (default: outputs/features).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video FPS (default: 25.0).",
    )

    args = parser.parse_args()

    extract_cluster_spatiotemporal_features(
        persistent_clusters_csv=args.persistent_clusters,
        person_clusters_csv=args.person_clusters,
        person_motion_csv=args.person_motion,
        output_dir=args.output_dir,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
