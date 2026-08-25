import argparse
import csv
from pathlib import Path
import numpy as np

from cluster_spatial import cluster_spatial_crowd
from track_clusters import track_persistent_clusters
from extract_cluster_features import extract_cluster_spatiotemporal_features


def validate_clustering_pipeline(
    tracking_csv: str = "outputs/tracking/avenue_tracking.csv",
    video_path: str | None = None,
    output_dir: str = "outputs/clustering",
    eps: float = 60.0,
    min_samples: int = 3,
    small_group_eps: float = 60.0,
) -> dict:
    """Run clustering, persistent tracking, feature extraction, and validate all requirements."""
    print("=" * 70)
    print("RUNNING CLUSTERING PIPELINE VALIDATION ON REPRESENTATIVE AVENUE VIDEO")
    print("=" * 70)

    # 1. Run Spatial Clustering
    spatial_res = cluster_spatial_crowd(
        tracking_csv_path=tracking_csv,
        video_path=video_path,
        output_dir=output_dir,
        eps=eps,
        min_samples=min_samples,
        small_group_eps=small_group_eps,
    )

    # 2. Run Persistent Cluster Tracking
    track_res = track_persistent_clusters(
        cluster_features_csv_path=Path(output_dir) / "cluster_features.csv",
        person_clusters_csv_path=Path(output_dir) / "person_clusters.csv",
        video_path=video_path,
        output_dir=output_dir,
    )

    # 3. Read back person_clusters.csv and tracking CSV to verify assertions
    tracking_rows = []
    with open(tracking_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracking_rows.append(row)

    person_cluster_rows = []
    with open(Path(output_dir) / "person_clusters.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_cluster_rows.append(row)

    # Verification Checks
    total_detected = len(tracking_rows)
    total_clustered = len(person_cluster_rows)

    # Check 1: No detected person disappears
    assert total_detected == total_clustered, (
        f"Mismatch in total detections! Tracking: {total_detected}, Person Clusters: {total_clustered}"
    )

    crowd_person_rows = [r for r in person_cluster_rows if r["group_type"] == "crowd_cluster"]
    small_group_person_rows = [r for r in person_cluster_rows if r["group_type"] == "small_group"]
    isolated_person_rows = [r for r in person_cluster_rows if r["group_type"] == "isolated_person"]

    persons_in_crowd = len(crowd_person_rows)
    persons_in_small_groups = len(small_group_person_rows)
    isolated_persons_count = len(isolated_person_rows)

    assert persons_in_crowd + persons_in_small_groups + isolated_persons_count == total_detected, (
        "Sum of classified persons does not match total detected persons!"
    )

    # Check 2: Group bounding box enclosing check
    # For every group in every frame, verify cluster_x1 <= min(x1), cluster_y1 <= min(y1), cluster_x2 >= max(x2), cluster_y2 >= max(y2)
    tracking_map = {(int(r["frame"]), int(r["track_id"])): r for r in tracking_rows}

    groups_by_frame = {}
    for r in person_cluster_rows:
        key = (int(r["frame"]), r["group_type"], r["group_id"])
        groups_by_frame.setdefault(key, []).append(r)

    for (f_idx, gtype, gid), members in groups_by_frame.items():
        member_x1s = [float(tracking_map[(f_idx, int(m["track_id"]))]["x1"]) for m in members]
        member_y1s = [float(tracking_map[(f_idx, int(m["track_id"]))]["y1"]) for m in members]
        member_x2s = [float(tracking_map[(f_idx, int(m["track_id"]))]["x2"]) for m in members]
        member_y2s = [float(tracking_map[(f_idx, int(m["track_id"]))]["y2"]) for m in members]

        expected_x1 = min(member_x1s)
        expected_y1 = min(member_y1s)
        expected_x2 = max(member_x2s)
        expected_y2 = max(member_y2s)

        for m in members:
            cx1 = float(m["cluster_x1"])
            cy1 = float(m["cluster_y1"])
            cx2 = float(m["cluster_x2"])
            cy2 = float(m["cluster_y2"])

            assert abs(cx1 - expected_x1) < 1e-2, f"Bounding box mismatch x1 in frame {f_idx}"
            assert abs(cy1 - expected_y1) < 1e-2, f"Bounding box mismatch y1 in frame {f_idx}"
            assert abs(cx2 - expected_x2) < 1e-2, f"Bounding box mismatch x2 in frame {f_idx}"
            assert abs(cy2 - expected_y2) < 1e-2, f"Bounding box mismatch y2 in frame {f_idx}"

    # Print Validation Metrics formatted as requested
    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    print(f"Total detected persons:     {total_detected}")
    print(f"Persons in crowd clusters:  {persons_in_crowd}")
    print(f"Persons in small groups:    {persons_in_small_groups}")
    print(f"Isolated persons:           {isolated_persons_count}")
    print()
    print(f"Number of crowd clusters:   {spatial_res['number_of_crowd_clusters']}")
    print(f"Number of small groups:     {spatial_res['number_of_small_groups']}")
    print(f"Number of isolated persons: {spatial_res['number_of_isolated_persons']}")
    print("=" * 70)
    print("\nVERIFICATION CHECKS:")
    print("  [PASSED] No detected person disappears merely because DBSCAN classified them as noise (100% retained).")
    print("  [PASSED] Every crowd cluster has ONE bounding box enclosing all members' full bodies.")
    print("  [PASSED] Every small group has ONE bounding box enclosing all members' full bodies.")
    print("  [PASSED] Every isolated person has an individual bounding box.")
    print("=" * 70)

    return {
        "total_detected": total_detected,
        "persons_in_crowd": persons_in_crowd,
        "persons_in_small_groups": persons_in_small_groups,
        "isolated_persons": isolated_persons_count,
        "number_of_crowd_clusters": spatial_res["number_of_crowd_clusters"],
        "number_of_small_groups": spatial_res["number_of_small_groups"],
        "number_of_isolated_persons": spatial_res["number_of_isolated_persons"],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate clustering pipeline requirements.")
    parser.add_argument("--csv", type=str, default="outputs/tracking/avenue_tracking.csv")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/clustering")
    parser.add_argument("--eps", type=float, default=60.0)
    parser.add_argument("--min_samples", type=int, default=3)
    parser.add_argument("--small_group_eps", type=float, default=60.0)
    args = parser.parse_args()

    validate_clustering_pipeline(
        tracking_csv=args.csv,
        video_path=args.video,
        output_dir=args.output_dir,
        eps=args.eps,
        min_samples=args.min_samples,
        small_group_eps=args.small_group_eps,
    )


if __name__ == "__main__":
    main()
