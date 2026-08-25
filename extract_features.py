import argparse
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def wrap_angle_deg(angle_diff: float) -> float:
    """Wrap angular difference into range [-180, 180] degrees."""
    return (angle_diff + 180.0) % 360.0 - 180.0


def extract_motion_features(
    tracking_csv_path: str | Path = "outputs/tracking/avenue_tracking.csv",
    output_dir: str | Path = "outputs/features",
    fps: float = 25.0,
    min_observations: int = 3,
    max_gap_frames: int = 10,
    jitter_threshold: float = 0.5,
) -> dict:
    """Extract individual frame-to-frame motion features and track-level summary statistics.

    Args:
        tracking_csv_path: Path to the input tracking CSV file.
        output_dir: Directory to save feature CSVs and diagnostic plots.
        fps: Video frames per second (time step delta_t = delta_frame / fps).
        min_observations: Minimum frame observations required for a track to be analyzed.
        max_gap_frames: Maximum allowable frame gap between consecutive detections in a track.
        jitter_threshold: Minimum displacement in pixels to consider directional movement.

    Returns:
        Dictionary containing summary statistics and file paths.
    """
    input_file = Path(tracking_csv_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Tracking CSV file not found: {input_file}")

    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame_features_path = out_dir / "person_motion_features.csv"
    track_summary_path = out_dir / "person_motion_summary.csv"

    # 1. Read and parse tracking CSV
    raw_tracks: dict[int, list[dict]] = {}
    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            obs = {
                "frame": int(row["frame"]),
                "track_id": tid,
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "foot_x": float(row["foot_x"]),
                "foot_y": float(row["foot_y"]),
                "confidence": float(row["confidence"]),
            }
            raw_tracks.setdefault(tid, []).append(obs)

    total_unique_tracks = len(raw_tracks)

    # 2. Sort observations chronologically and filter by min_observations
    all_frame_features = []
    track_summaries = []

    for tid, observations in sorted(raw_tracks.items()):
        # Sort observations by frame
        observations.sort(key=lambda x: x["frame"])

        # Filter out tracks with too few observations
        if len(observations) < min_observations:
            continue

        prev_obs = None
        prev_vx = None
        prev_vy = None
        prev_direction = None

        track_speeds = []
        track_accels = []
        track_dir_changes = []
        track_step_displacements = []
        track_widths = []
        track_heights = []
        track_areas = []

        for i, obs in enumerate(observations):
            curr_frame = obs["frame"]
            curr_foot_x = obs["foot_x"]
            curr_foot_y = obs["foot_y"]
            bbox_w = obs["width"]
            bbox_h = obs["height"]
            bbox_area = bbox_w * bbox_h

            track_widths.append(bbox_w)
            track_heights.append(bbox_h)
            track_areas.append(bbox_area)

            if prev_obs is None:
                # First observation of track
                dx = 0.0
                dy = 0.0
                disp = 0.0
                vx = 0.0
                vy = 0.0
                speed = 0.0
                accel = 0.0
                direction_deg = ""
                direction_change_deg = 0.0
            else:
                frame_diff = curr_frame - prev_obs["frame"]
                if frame_diff <= 0:
                    continue  # Guard against duplicate frame numbers

                # Time elapsed in seconds
                dt = frame_diff / fps

                if frame_diff > max_gap_frames:
                    # Tracking gap exceeded: calculate gap velocity, reset acceleration/direction reference
                    dx = curr_foot_x - prev_obs["foot_x"]
                    dy = curr_foot_y - prev_obs["foot_y"]
                    disp = math.sqrt(dx * dx + dy * dy)
                    vx = dx / dt
                    vy = dy / dt
                    speed = disp / dt
                    accel = 0.0
                    direction_deg = math.degrees(math.atan2(dy, dx)) if disp >= jitter_threshold else ""
                    direction_change_deg = 0.0
                    prev_vx = vx
                    prev_vy = vy
                    prev_direction = direction_deg if direction_deg != "" else None
                else:
                    dx = curr_foot_x - prev_obs["foot_x"]
                    dy = curr_foot_y - prev_obs["foot_y"]
                    disp = math.sqrt(dx * dx + dy * dy)
                    vx = dx / dt
                    vy = dy / dt
                    speed = disp / dt

                    # Acceleration: vector acceleration magnitude (|a| = sqrt(ax^2 + ay^2))
                    if prev_vx is not None and prev_vy is not None:
                        ax = (vx - prev_vx) / dt
                        ay = (vy - prev_vy) / dt
                        accel = math.sqrt(ax * ax + ay * ay)
                        track_accels.append(accel)
                    else:
                        accel = 0.0

                    # Movement direction in degrees [-180, 180]
                    if disp >= jitter_threshold:
                        curr_direction = math.degrees(math.atan2(dy, dx))
                        direction_deg = f"{curr_direction:.2f}"

                        # Direction change with angle wrapping
                        if prev_direction is not None and prev_direction != "":
                            raw_diff = curr_direction - float(prev_direction)
                            wrapped_diff = abs(wrap_angle_deg(raw_diff))
                            direction_change_deg = wrapped_diff
                            track_dir_changes.append(direction_change_deg)
                        else:
                            direction_change_deg = 0.0

                        prev_direction = curr_direction
                    else:
                        # Stationary jitter below threshold
                        direction_deg = ""
                        direction_change_deg = 0.0

                    prev_vx = vx
                    prev_vy = vy

                track_speeds.append(speed)
                track_step_displacements.append(disp)

            prev_obs = obs

            # Append frame-level feature record
            all_frame_features.append({
                "frame": curr_frame,
                "track_id": tid,
                "foot_x": f"{curr_foot_x:.2f}",
                "foot_y": f"{curr_foot_y:.2f}",
                "dx": f"{dx:.2f}",
                "dy": f"{dy:.2f}",
                "displacement": f"{disp:.2f}",
                "velocity_x": f"{vx:.2f}",
                "velocity_y": f"{vy:.2f}",
                "speed": f"{speed:.2f}",
                "acceleration": f"{accel:.2f}",
                "direction": direction_deg,
                "direction_change": f"{direction_change_deg:.2f}",
                "bbox_width": f"{bbox_w:.2f}",
                "bbox_height": f"{bbox_h:.2f}",
                "bbox_area": f"{bbox_area:.2f}",
            })

        # Calculate track summary metrics
        first_obs = observations[0]
        last_obs = observations[-1]
        track_duration_sec = (last_obs["frame"] - first_obs["frame"]) / fps
        obs_count = len(observations)
        trajectory_length = sum(track_step_displacements)

        total_net_disp = math.sqrt(
            (last_obs["foot_x"] - first_obs["foot_x"]) ** 2
            + (last_obs["foot_y"] - first_obs["foot_y"]) ** 2
        )

        mean_speed = float(np.mean(track_speeds)) if track_speeds else 0.0
        median_speed = float(np.median(track_speeds)) if track_speeds else 0.0
        speed_std = float(np.std(track_speeds)) if track_speeds else 0.0
        max_speed = float(np.max(track_speeds)) if track_speeds else 0.0

        mean_accel = float(np.mean(track_accels)) if track_accels else 0.0
        max_accel = float(np.max(track_accels)) if track_accels else 0.0

        mean_dir_change = float(np.mean(track_dir_changes)) if track_dir_changes else 0.0
        max_dir_change = float(np.max(track_dir_changes)) if track_dir_changes else 0.0

        mean_w = float(np.mean(track_widths))
        mean_h = float(np.mean(track_heights))
        mean_area = float(np.mean(track_areas))

        track_summaries.append({
            "track_id": tid,
            "track_duration": f"{track_duration_sec:.2f}",
            "observation_count": obs_count,
            "trajectory_length": f"{trajectory_length:.2f}",
            "total_displacement": f"{total_net_disp:.2f}",
            "mean_speed": f"{mean_speed:.2f}",
            "median_speed": f"{median_speed:.2f}",
            "speed_std": f"{speed_std:.2f}",
            "max_speed": f"{max_speed:.2f}",
            "mean_acceleration": f"{mean_accel:.2f}",
            "max_acceleration": f"{max_accel:.2f}",
            "mean_direction_change": f"{mean_dir_change:.2f}",
            "max_direction_change": f"{max_dir_change:.2f}",
            "mean_bbox_width": f"{mean_w:.2f}",
            "mean_bbox_height": f"{mean_h:.2f}",
            "mean_bbox_area": f"{mean_area:.2f}",
        })

    # 3. Write Output 1: Frame-level features CSV
    frame_cols = [
        "frame",
        "track_id",
        "foot_x",
        "foot_y",
        "dx",
        "dy",
        "displacement",
        "velocity_x",
        "velocity_y",
        "speed",
        "acceleration",
        "direction",
        "direction_change",
        "bbox_width",
        "bbox_height",
        "bbox_area",
    ]
    with open(frame_features_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_cols)
        writer.writeheader()
        writer.writerows(all_frame_features)

    # 4. Write Output 2: Track-level summary CSV
    summary_cols = [
        "track_id",
        "track_duration",
        "observation_count",
        "trajectory_length",
        "total_displacement",
        "mean_speed",
        "median_speed",
        "speed_std",
        "max_speed",
        "mean_acceleration",
        "max_acceleration",
        "mean_direction_change",
        "max_direction_change",
        "mean_bbox_width",
        "mean_bbox_height",
        "mean_bbox_area",
    ]
    with open(track_summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(track_summaries)

    # 5. Generate Output 3: Diagnostic Visualizations
    valid_tracks_count = len(track_summaries)
    all_speeds = [float(row["speed"]) for row in all_frame_features if float(row["speed"]) > 0]
    all_accels = [float(row["acceleration"]) for row in all_frame_features if float(row["acceleration"]) > 0]
    all_dir_changes = [float(row["direction_change"]) for row in all_frame_features if float(row["direction_change"]) > 0]

    # Plot 1: Speed Distribution
    plt.figure(figsize=(8, 5))
    if all_speeds:
        plt.hist(all_speeds, bins=30, color="#2563EB", edgecolor="black", alpha=0.75)
        plt.axvline(np.mean(all_speeds), color="#DC2626", linestyle="--", label=f"Mean: {np.mean(all_speeds):.1f} px/s")
        plt.axvline(np.median(all_speeds), color="#16A34A", linestyle=":", label=f"Median: {np.median(all_speeds):.1f} px/s")
    plt.title("Person Speed Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Speed (pixels/second)")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "speed_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Acceleration Distribution
    plt.figure(figsize=(8, 5))
    if all_accels:
        plt.hist(all_accels, bins=30, color="#7C3AED", edgecolor="black", alpha=0.75)
        plt.axvline(np.mean(all_accels), color="#DC2626", linestyle="--", label=f"Mean: {np.mean(all_accels):.1f} px/s²")
        plt.axvline(np.median(all_accels), color="#16A34A", linestyle=":", label=f"Median: {np.median(all_accels):.1f} px/s²")
    plt.title("Person Acceleration Magnitude Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Acceleration Magnitude (pixels/second²)")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "acceleration_distribution.png", dpi=150)
    plt.close()

    # Plot 3: Direction-Change Distribution
    plt.figure(figsize=(8, 5))
    if all_dir_changes:
        plt.hist(all_dir_changes, bins=30, color="#EA580C", edgecolor="black", alpha=0.75)
        plt.axvline(np.mean(all_dir_changes), color="#DC2626", linestyle="--", label=f"Mean: {np.mean(all_dir_changes):.1f}°")
    plt.title("Direction Change Distribution (Angular Deviation)", fontsize=13, fontweight="bold")
    plt.xlabel("Direction Change (degrees, [0° - 180°])")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "direction_change_distribution.png", dpi=150)
    plt.close()

    # Plot 4: Representative Trajectories in Image Coordinates
    plt.figure(figsize=(10, 6))
    # Select up to 10 longest/representative tracks
    sorted_by_len = sorted(track_summaries, key=lambda x: int(x["observation_count"]), reverse=True)
    selected_tids = [s["track_id"] for s in sorted_by_len[:10]]

    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(10, len(selected_tids))))

    for idx, tid in enumerate(selected_tids):
        t_obs = raw_tracks[tid]
        fx = [o["foot_x"] for o in t_obs]
        fy = [o["foot_y"] for o in t_obs]
        c = color_cycle[idx % len(color_cycle)]
        plt.plot(fx, fy, "-o", markersize=3, label=f"Track ID {tid} ({len(fx)} pts)", color=c, alpha=0.85)
        plt.plot(fx[0], fy[0], "s", markersize=7, color=c, markeredgecolor="black")  # Start
        plt.plot(fx[-1], fy[-1], "X", markersize=8, color=c, markeredgecolor="black")  # End

    plt.title("Representative Person Foot-Point Trajectories (Image Space: 640x360)", fontsize=13, fontweight="bold")
    plt.xlabel("Image X Coordinate (pixels)")
    plt.ylabel("Image Y Coordinate (pixels)")
    plt.xlim(0, 640)
    plt.ylim(360, 0)  # Inverted Y for camera coordinates
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "trajectories_plot.png", dpi=150)
    plt.close()

    # Aggregate Statistics
    overall_avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    overall_max_speed = float(np.max(all_speeds)) if all_speeds else 0.0
    overall_avg_accel = float(np.mean(all_accels)) if all_accels else 0.0
    overall_max_accel = float(np.max(all_accels)) if all_accels else 0.0

    print("=" * 60)
    print("Motion Feature Extraction Summary")
    print("=" * 60)
    print(f"Number of unique tracks:             {total_unique_tracks}")
    print(f"Number of valid tracks after filter: {valid_tracks_count}")
    print(f"Total feature rows:                  {len(all_frame_features)}")
    print(f"Source FPS:                          {fps:.2f}")
    print(f"Average speed:                       {overall_avg_speed:.2f} pixels/second")
    print(f"Maximum speed:                       {overall_max_speed:.2f} pixels/second")
    print(f"Average acceleration:                {overall_avg_accel:.2f} pixels/second^2")
    print(f"Maximum acceleration:                {overall_max_accel:.2f} pixels/second^2")
    print(f"Frame features CSV:                  {frame_features_path}")
    print(f"Track summary CSV:                   {track_summary_path}")
    print(f"Diagnostic plots directory:          {plots_dir}")
    print("=" * 60)

    return {
        "unique_tracks": total_unique_tracks,
        "valid_tracks": valid_tracks_count,
        "total_feature_rows": len(all_frame_features),
        "source_fps": fps,
        "average_speed": overall_avg_speed,
        "max_speed": overall_max_speed,
        "average_acceleration": overall_avg_accel,
        "max_acceleration": overall_max_accel,
        "frame_features_csv": str(frame_features_path),
        "track_summary_csv": str(track_summary_path),
        "plots_dir": str(plots_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract handcrafted individual and track-level motion features from tracking CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/tracking/avenue_tracking.csv",
        help="Path to input tracking CSV (default: outputs/tracking/avenue_tracking.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/features",
        help="Directory to save features and plots (default: outputs/features).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Source video FPS for velocity/acceleration timing (default: 25.0).",
    )
    parser.add_argument(
        "--min_obs",
        type=int,
        default=3,
        help="Minimum observations required to retain track (default: 3).",
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=10,
        help="Maximum allowable frame gap between consecutive detections (default: 10).",
    )

    args = parser.parse_args()

    extract_motion_features(
        tracking_csv_path=args.csv,
        output_dir=args.output_dir,
        fps=args.fps,
        min_observations=args.min_obs,
        max_gap_frames=args.max_gap,
    )


if __name__ == "__main__":
    main()
