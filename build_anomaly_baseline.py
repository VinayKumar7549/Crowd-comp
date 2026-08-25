import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
import sklearn


# ==============================================================================
# PART A: FEATURE QUALITY AUDIT
# ==============================================================================

def run_feature_quality_audit(
    feature_rows: list[dict],
    numerical_cols: list[str],
    output_dir: Path,
) -> tuple[list[dict], dict]:
    """Audit all candidate numerical behavioral features for validity, missingness, and range."""
    audit_records = []
    distributions = {}

    for col in numerical_cols:
        raw_vals = [r.get(col) for r in feature_rows]
        missing_count = sum(1 for v in raw_vals if v is None or v == "")

        parsed_vals = []
        inf_count = 0
        zero_count = 0
        negative_count = 0

        for v in raw_vals:
            if v is not None and v != "":
                try:
                    f_val = float(v)
                    if math.isinf(f_val):
                        inf_count += 1
                    elif math.isnan(f_val):
                        missing_count += 1
                    else:
                        parsed_vals.append(f_val)
                        if f_val == 0.0:
                            zero_count += 1
                        elif f_val < 0.0:
                            negative_count += 1
                except ValueError:
                    missing_count += 1

        if parsed_vals:
            arr = np.array(parsed_vals, dtype=float)
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            mean_v = float(np.mean(arr))
            median_v = float(np.median(arr))
            std_v = float(np.std(arr))
            p01 = float(np.percentile(arr, 1))
            p05 = float(np.percentile(arr, 5))
            p25 = float(np.percentile(arr, 25))
            p50 = float(np.percentile(arr, 50))
            p75 = float(np.percentile(arr, 75))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            distributions[col] = arr
        else:
            min_v = max_v = mean_v = median_v = std_v = 0.0
            p01 = p05 = p25 = p50 = p75 = p95 = p99 = 0.0

        audit_records.append({
            "feature": col,
            "dtype": "float64" if parsed_vals else "unknown",
            "missing_count": missing_count,
            "infinite_count": inf_count,
            "zero_count": zero_count,
            "negative_count": negative_count,
            "minimum": f"{min_v:.4f}",
            "maximum": f"{max_v:.4f}",
            "mean": f"{mean_v:.4f}",
            "median": f"{median_v:.4f}",
            "std": f"{std_v:.4f}",
            "p01": f"{p01:.4f}",
            "p05": f"{p05:.4f}",
            "p25": f"{p25:.4f}",
            "p50": f"{p50:.4f}",
            "p75": f"{p75:.4f}",
            "p95": f"{p95:.4f}",
            "p99": f"{p99:.4f}",
        })

    # Save feature_audit.csv
    audit_csv_path = output_dir / "feature_audit.csv"
    with open(audit_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fields = [
            "feature", "dtype", "missing_count", "infinite_count", "zero_count",
            "negative_count", "minimum", "maximum", "mean", "median", "std",
            "p01", "p05", "p25", "p50", "p75", "p95", "p99",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_records)

    # Plot multi-panel feature audit distributions
    major_feats = [
        ("cluster_speed", "Cluster Speed (px/s)"),
        ("cluster_acceleration", "Cluster Acceleration (px/s²)"),
        ("density", "Density (people/px²)"),
        ("directional_coherence", "Directional Coherence [0, 1]"),
        ("spatial_dispersion", "Spatial Dispersion (px)"),
        ("area_change", "Area Change Rate (px²/s)"),
        ("density_change", "Density Change Rate"),
        ("cluster_direction_change", "Direction Change (°)"),
        ("number_of_people", "Number of People"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (f_key, f_title) in enumerate(major_feats):
        ax = axes[idx]
        if f_key in distributions:
            data = distributions[f_key]
            # Clip display range for extreme skew in speed/acceleration/area_change for visual clarity
            p99_val = np.percentile(data, 99)
            if f_key in ["cluster_speed", "cluster_acceleration", "area_change", "density_change"]:
                p01_val = np.percentile(data, 1)
                vis_data = data[(data >= p01_val) & (data <= p99_val)]
            else:
                vis_data = data

            ax.hist(vis_data, bins=25, color="#2563EB", edgecolor="black", alpha=0.75)
            ax.axvline(np.mean(data), color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(data):.1f}")
            ax.axvline(np.median(data), color="#16A34A", linestyle=":", linewidth=1.5, label=f"Med: {np.median(data):.1f}")
            ax.set_title(f_title, fontsize=10, fontweight="bold")
            ax.set_ylabel("Count", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    plt.suptitle("Avenue Normal Training Feature Quality Audit Distributions", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    audit_png_path = output_dir / "feature_audit.png"
    plt.savefig(audit_png_path, dpi=150)
    plt.close()

    return audit_records, distributions


# ==============================================================================
# PART B: IDENTIFY POTENTIAL TRACKING ARTIFACTS / EXTREME OBSERVATIONS
# ==============================================================================

def investigate_extreme_observations(
    feature_rows: list[dict],
    distributions: dict[str, np.ndarray],
    output_dir: Path,
) -> list[dict]:
    """Identify and categorize the top 1% observations across volatile motion and area derivatives."""
    extreme_records = []
    features_to_investigate = [
        "cluster_speed",
        "cluster_acceleration",
        "area_change",
        "density_change",
        "cluster_direction_change",
    ]

    thresholds = {}
    for feat in features_to_investigate:
        if feat in distributions:
            arr = distributions[feat]
            p99 = float(np.percentile(arr, 99))
            p01 = float(np.percentile(arr, 1))
            thresholds[feat] = (p01, p99)

    for row in feature_rows:
        v_id = row["video_id"]
        f_idx = int(row["frame"])
        pid = int(row["persistent_cluster_id"])

        for feat in features_to_investigate:
            if feat in thresholds and row.get(feat) not in ["", None]:
                val = float(row[feat])
                p01, p99 = thresholds[feat]

                is_extreme = False
                if feat in ["area_change", "density_change"]:
                    if val >= p99 or val <= p01:
                        is_extreme = True
                else:
                    if val >= p99:
                        is_extreme = True

                if is_extreme:
                    # Categorize potential physical / algorithmic source
                    if feat == "cluster_speed":
                        if val > 1500.0:
                            cat = "Centroid association jump / member track swap"
                        else:
                            cat = "Rapid camera-perspective motion / brisk transit"
                    elif feat == "cluster_acceleration":
                        if val > 50000.0:
                            cat = "Centroid derivative spike across discrete frame step"
                        else:
                            cat = "Abrupt onset/cessation of pedestrian group movement"
                    elif feat == "area_change":
                        if abs(val) > 100000.0:
                            cat = "Bounding box footprint expansion/contraction jump"
                        else:
                            cat = "Pedestrian entering/leaving spatial DBSCAN cluster"
                    elif feat == "density_change":
                        cat = "Rapid density adjustment due to boundary member change"
                    elif feat == "cluster_direction_change":
                        cat = "Sharp cluster heading shift / turnaround or jitter wrap"
                    else:
                        cat = "Unusual statistical outlier"

                    extreme_records.append({
                        "video_id": v_id,
                        "frame": f_idx,
                        "persistent_cluster_id": pid,
                        "feature_name": feat,
                        "feature_value": f"{val:.4f}",
                        "percentile_threshold": f"{p99:.4f}" if val >= p99 else f"{p01:.4f}",
                        "potential_artifact_type": cat,
                    })

    extreme_csv_path = output_dir / "extreme_observations.csv"
    with open(extreme_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fields = [
            "video_id", "frame", "persistent_cluster_id", "feature_name",
            "feature_value", "percentile_threshold", "potential_artifact_type",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(extreme_records)

    return extreme_records


# ==============================================================================
# PART C: FEATURE SELECTION
# ==============================================================================

def define_feature_selection(output_dir: Path) -> dict:
    """Document candidate selected features, excluded features, and rationale for anomaly modeling."""
    selected_features = [
        {
            "name": "cluster_speed",
            "unit": "pixels / second",
            "definition": "Instantaneous Euclidean velocity magnitude of the crowd cluster centroid.",
            "behavioral_significance": "Distinguishes running, swift rushing, or abrupt stopping from normal walking.",
        },
        {
            "name": "cluster_acceleration",
            "unit": "pixels / second²",
            "definition": "Vector acceleration magnitude of the crowd cluster centroid.",
            "behavioral_significance": "Captures sudden surges, panic onset, or violent halts.",
        },
        {
            "name": "density",
            "unit": "people / pixel²",
            "definition": "Ratio of cluster member count to bounding box ground area.",
            "behavioral_significance": "Identifies dangerous crowd compression, crushing, or high congestion.",
        },
        {
            "name": "directional_coherence",
            "unit": "normalized index [0.0, 1.0]",
            "definition": "Magnitude of the mean member unit velocity vectors.",
            "behavioral_significance": "Measures group alignment; low values indicate conflicting movements or panic scattering.",
        },
        {
            "name": "spatial_dispersion",
            "unit": "pixels",
            "definition": "Root Mean Square Distance (RMSD) of member pedestrian foot points to centroid.",
            "behavioral_significance": "Quantifies spatial tightness and formation spreading.",
        },
        {
            "name": "mean_individual_speed",
            "unit": "pixels / second",
            "definition": "Mean speed of individual pedestrians within the cluster.",
            "behavioral_significance": "Differentiates running individuals within a slowly advancing group.",
        },
        {
            "name": "velocity_dispersion",
            "unit": "pixels / second",
            "definition": "Standard deviation of member velocity vectors around the group mean velocity.",
            "behavioral_significance": "Highlights internal turbulence and non-uniform member kinetics.",
        },
        {
            "name": "area_change",
            "unit": "pixels² / second",
            "definition": "Temporal rate of change in cluster bounding box area.",
            "behavioral_significance": "Detects explosive dispersion (e.g. bomb panic) or sudden gathering.",
        },
        {
            "name": "density_change",
            "unit": "people / (pixel² · second)",
            "definition": "Temporal rate of change in spatial crowd density.",
            "behavioral_significance": "Signals rapid crowding or sudden emptying of a walkway.",
        },
        {
            "name": "cluster_direction_change",
            "unit": "degrees [0, 180]",
            "definition": "Absolute wrapped angular deviation between consecutive centroid velocity vectors.",
            "behavioral_significance": "Captures sudden direction reversals, erratic pacing, or evasive turns.",
        },
        {
            "name": "number_of_people",
            "unit": "count",
            "definition": "Number of detected pedestrians comprising the spatial cluster.",
            "behavioral_significance": "Establishes crowd scale context for density and dispersion.",
        },
    ]

    excluded_features = [
        {"name": "video_id", "reason": "Identifier string; non-behavioral."},
        {"name": "frame", "reason": "Temporal index; non-behavioral."},
        {"name": "persistent_cluster_id", "reason": "Tracking identity integer; non-behavioral."},
        {"name": "centroid_x", "reason": "Raw image position coordinate; camera-perspective dependent."},
        {"name": "centroid_y", "reason": "Raw image position coordinate; camera-perspective dependent."},
        {"name": "centroid_dx", "reason": "Raw displacement components; redundant with cluster_velocity and cluster_speed."},
        {"name": "centroid_dy", "reason": "Raw displacement components; redundant with cluster_velocity and cluster_speed."},
        {"name": "cluster_velocity_x", "reason": "Directional velocity component; camera-axis dependent. Coherence and direction change capture coordinate-invariant dynamics."},
        {"name": "cluster_velocity_y", "reason": "Directional velocity component; camera-axis dependent. Coherence and direction change capture coordinate-invariant dynamics."},
        {"name": "mean_velocity_x", "reason": "Directional velocity component; camera-axis dependent."},
        {"name": "mean_velocity_y", "reason": "Directional velocity component; camera-axis dependent."},
        {"name": "cluster_displacement", "reason": "Redundant with cluster_speed (proportional by FPS)."},
        {"name": "cluster_direction", "reason": "Absolute heading angle is camera-dependent; cluster_direction_change is the rotation-invariant behavioral metric."},
        {"name": "cluster_width", "reason": "Redundant with cluster_area and spatial_dispersion."},
        {"name": "cluster_height", "reason": "Redundant with cluster_area and spatial_dispersion."},
        {"name": "width_change", "reason": "Redundant with area_change."},
        {"name": "height_change", "reason": "Redundant with area_change."},
        {"name": "std_individual_speed", "reason": "Subsumed by velocity_dispersion."},
        {"name": "max_individual_speed", "reason": "Highly correlated with mean_individual_speed."},
        {"name": "mean_distance_to_centroid", "reason": "Subsumed by RMSD spatial_dispersion."},
        {"name": "median_distance_to_centroid", "reason": "Subsumed by RMSD spatial_dispersion."},
        {"name": "std_distance_to_centroid", "reason": "Subsumed by RMSD spatial_dispersion."},
        {"name": "max_distance_to_centroid", "reason": "Subsumed by RMSD spatial_dispersion."},
    ]

    selected_feature_names = [f["name"] for f in selected_features]

    spec = {
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_feature_names": selected_feature_names,
        "excluded_feature_count": len(excluded_features),
        "excluded_features": excluded_features,
    }

    with open(output_dir / "selected_features.json", mode="w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    return spec


# ==============================================================================
# PART D: VIDEO-LEVEL SPLITTING
# ==============================================================================

def create_video_level_split(
    feature_rows: list[dict],
    validation_ratio: float = 0.25,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Perform deterministic video-level train/validation split to avoid temporal autocorrelation leakage."""
    video_ids = sorted(list(set(r["video_id"] for r in feature_rows)))
    num_total_vids = len(video_ids)

    # Deterministic split: Hold out 4 videos across representative lengths
    # Avenue dataset has 16 training videos.
    # Selecting video_04 (540 rows), video_08 (962 rows), video_12 (380 rows), video_16 (632 rows)
    # provides a balanced distribution across early, middle, and late recording sessions (2,514 rows = 17.0%).
    val_vids = ["video_04", "video_08", "video_12", "video_16"]
    train_vids = [v for v in video_ids if v not in val_vids]

    train_rows = [r for r in feature_rows if r["video_id"] in train_vids]
    val_rows = [r for r in feature_rows if r["video_id"] in val_vids]

    split_meta = {
        "split_strategy": "Deterministic Video-Level Stratified Holdout (0 temporal frame leakage)",
        "random_seed": random_seed,
        "total_videos": num_total_vids,
        "training_videos": train_vids,
        "validation_videos": val_vids,
        "training_videos_count": len(train_vids),
        "validation_videos_count": len(val_vids),
        "training_rows_count": len(train_rows),
        "validation_rows_count": len(val_rows),
        "validation_row_percentage": f"{(len(val_rows) / len(feature_rows) * 100.0):.2f}%",
    }

    if output_dir:
        with open(output_dir / "split_metadata.json", mode="w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

    return train_rows, val_rows, split_meta


# ==============================================================================
# PART E: NORMAL FEATURE PREPROCESSING & MATRIX BUILDER
# ==============================================================================

def preprocess_features(
    rows: list[dict],
    feature_names: list[str],
    train_medians: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Convert row dictionaries to clean numerical numpy matrix with explicit imputation and quality audit."""
    if train_medians is None:
        # Calculate medians from training split
        train_medians = {}
        for f_name in feature_names:
            valid_vals = []
            for r in rows:
                v = r.get(f_name)
                if v is not None and v != "":
                    try:
                        fv = float(v)
                        if not math.isnan(fv) and not math.isinf(fv):
                            valid_vals.append(fv)
                    except ValueError:
                        pass
            train_medians[f_name] = float(np.median(valid_vals)) if valid_vals else 0.0

    X_list = []
    imputed_count = 0

    for r in rows:
        feat_vector = []
        for f_name in feature_names:
            v = r.get(f_name)
            if v is None or v == "":
                feat_vector.append(train_medians[f_name])
                imputed_count += 1
            else:
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        feat_vector.append(train_medians[f_name])
                        imputed_count += 1
                    else:
                        feat_vector.append(fv)
                except ValueError:
                    feat_vector.append(train_medians[f_name])
                    imputed_count += 1
        X_list.append(feat_vector)

    X = np.array(X_list, dtype=np.float64)

    prep_info = {
        "imputed_values_count": imputed_count,
        "feature_medians": train_medians,
        "sample_count": len(rows),
        "feature_count": len(feature_names),
        "scaling": "None (Tree-based Isolation Forest is invariant to monotonic feature scaling)",
    }

    return X, train_medians, prep_info


# ==============================================================================
# PART I: TEMPORAL SMOOTHING UTILITIES
# ==============================================================================

def apply_temporal_smoothing(
    scores: np.ndarray,
    method: str = "moving_average",
    window_size: int = 5,
) -> np.ndarray:
    """Apply temporal smoothing over consecutive anomaly score observations."""
    if len(scores) < window_size:
        return scores.copy()

    if method == "moving_average":
        kernel = np.ones(window_size) / window_size
        # Pad edges
        pad_size = window_size // 2
        padded = np.pad(scores, pad_size, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[:len(scores)]

    elif method == "median_filter":
        pad_size = window_size // 2
        padded = np.pad(scores, pad_size, mode="edge")
        smoothed = np.zeros_like(scores)
        for i in range(len(scores)):
            smoothed[i] = np.median(padded[i : i + window_size])
        return smoothed

    else:
        return scores.copy()


# ==============================================================================
# MAIN ANOMALY BASELINE PIPELINE
# ==============================================================================

def build_handcrafted_anomaly_baseline(
    merged_features_csv: str | Path = "outputs/training_dataset/normal_cluster_features.csv",
    output_dir: str | Path = "outputs/anomaly_baseline",
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    contamination: str | float = "auto",
    random_state: int = 42,
) -> dict:
    """End-to-end handcrafted anomaly baseline workflow."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(merged_features_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Normal feature dataset not found: {csv_path}")

    # Load normal cluster features
    print("=" * 70)
    print("HANDCRAFTED ANOMALY DETECTION BASELINE PIPELINE")
    print("=" * 70)
    print(f"Loading normal training feature dataset from: {csv_path}")

    all_rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames
        all_rows = list(reader)

    print(f"Loaded {len(all_rows)} total cluster feature rows across {len(set(r['video_id'] for r in all_rows))} videos.")

    # --------------------------------------------------------------------------
    # PART A: Feature Quality Audit
    # --------------------------------------------------------------------------
    print("\n[PART A] Executing Feature Quality Audit...")
    num_cols = [c for c in all_cols if c not in ["video_id", "frame", "persistent_cluster_id", "cluster_direction"]]
    audit_records, distributions = run_feature_quality_audit(all_rows, num_cols, out_dir)
    print(f"  Audit complete. Saved audit table to: {out_dir / 'feature_audit.csv'}")
    print(f"  Saved multi-panel distribution plot to: {out_dir / 'feature_audit.png'}")

    # --------------------------------------------------------------------------
    # PART B: Identify Potential Tracking Artifacts
    # --------------------------------------------------------------------------
    print("\n[PART B] Investigating Extreme 1% Observations & Tracking Artifacts...")
    extreme_records = investigate_extreme_observations(all_rows, distributions, out_dir)
    print(f"  Identified {len(extreme_records)} extreme observation records across derivative/velocity features.")
    print(f"  Saved extreme observations report to: {out_dir / 'extreme_observations.csv'}")

    # --------------------------------------------------------------------------
    # PART C: Feature Selection
    # --------------------------------------------------------------------------
    print("\n[PART C] Defining Candidate Feature Selection...")
    feat_spec = define_feature_selection(out_dir)
    selected_feature_names = feat_spec["selected_feature_names"]
    print(f"  Selected {len(selected_feature_names)} behaviorally meaningful features: {selected_feature_names}")
    print(f"  Excluded {feat_spec['excluded_feature_count']} raw coordinates and redundant derivatives.")
    print(f"  Saved feature selection spec to: {out_dir / 'selected_features.json'}")

    # --------------------------------------------------------------------------
    # PART D: Video-Level Data Split
    # --------------------------------------------------------------------------
    print("\n[PART D] Creating Video-Level Train / Validation Split...")
    train_rows, val_rows, split_meta = create_video_level_split(all_rows, random_seed=random_state, output_dir=out_dir)
    print(f"  Training Split:   {split_meta['training_videos_count']} videos ({split_meta['training_rows_count']} feature rows)")
    print(f"  Validation Split: {split_meta['validation_videos_count']} videos ({split_meta['validation_rows_count']} feature rows)")
    print(f"  Saved split metadata to: {out_dir / 'split_metadata.json'}")

    # --------------------------------------------------------------------------
    # PART E: Preprocessing
    # --------------------------------------------------------------------------
    print("\n[PART E] Preprocessing Feature Matrices...")
    X_train, train_medians, prep_train = preprocess_features(train_rows, selected_feature_names)
    X_val, _, prep_val = preprocess_features(val_rows, selected_feature_names, train_medians=train_medians)

    prep_config = {
        "selected_features": selected_feature_names,
        "train_medians": train_medians,
        "train_imputed_count": prep_train["imputed_values_count"],
        "val_imputed_count": prep_val["imputed_values_count"],
        "scaling_applied": "None (Tree-based model)",
    }
    with open(out_dir / "preprocessing.json", mode="w", encoding="utf-8") as f:
        json.dump(prep_config, f, indent=2)
    print(f"  Preprocessing complete. Imputed train values: {prep_train['imputed_values_count']}, val values: {prep_val['imputed_values_count']}")
    print(f"  Saved preprocessing config to: {out_dir / 'preprocessing.json'}")

    # --------------------------------------------------------------------------
    # PART F: Isolation Forest Baseline Fitting (Train Split ONLY)
    # --------------------------------------------------------------------------
    print("\n[PART F] Fitting Isolation Forest on Training Split...")
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)
    print(f"  Isolation Forest fitted successfully with {n_estimators} estimators (contamination={contamination}).")

    # Save model artifact
    model_artifact_path = out_dir / "isolation_forest.joblib"
    joblib.dump(iso_forest, model_artifact_path)
    print(f"  Saved model artifact to: {model_artifact_path}")

    # --------------------------------------------------------------------------
    # PART G: Normal Validation Scoring (Validation Split ONLY)
    # --------------------------------------------------------------------------
    print("\n[PART G] Evaluating Model on Held-Out Normal Validation Split...")
    # sklearn score_samples returns opposite of anomaly score (lower is more anomalous).
    # We define: anomaly_score = - score_samples(X), so higher score = higher abnormality.
    val_raw_scores = iso_forest.score_samples(X_val)
    val_anomaly_scores = -val_raw_scores
    val_preds = iso_forest.predict(X_val)  # +1 inlier, -1 outlier

    val_score_records = []
    for r, score, pred in zip(val_rows, val_anomaly_scores, val_preds):
        val_score_records.append({
            "video_id": r["video_id"],
            "frame": r["frame"],
            "persistent_cluster_id": r["persistent_cluster_id"],
            "anomaly_score": f"{score:.6f}",
            "model_prediction": int(pred),
        })

    val_scores_csv_path = out_dir / "normal_validation_scores.csv"
    with open(val_scores_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fields = ["video_id", "frame", "persistent_cluster_id", "anomaly_score", "model_prediction"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(val_score_records)

    # Validation Score Distribution Plot
    plt.figure(figsize=(8, 5))
    plt.hist(val_anomaly_scores, bins=35, color="#2563EB", edgecolor="black", alpha=0.75)
    mean_s = float(np.mean(val_anomaly_scores))
    med_s = float(np.median(val_anomaly_scores))
    p95_s = float(np.percentile(val_anomaly_scores, 95))
    p99_s = float(np.percentile(val_anomaly_scores, 99))

    plt.axvline(mean_s, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_s:.4f}")
    plt.axvline(med_s, color="#16A34A", linestyle=":", linewidth=1.5, label=f"Median: {med_s:.4f}")
    plt.axvline(p95_s, color="#EA580C", linestyle="-.", linewidth=1.5, label=f"95th %ile: {p95_s:.4f}")
    plt.axvline(p99_s, color="#9333EA", linestyle="-", linewidth=1.5, label=f"99th %ile: {p99_s:.4f}")

    plt.title("Normal Validation Anomaly Score Distribution (Isolation Forest)", fontsize=12, fontweight="bold")
    plt.xlabel("Anomaly Score (Higher = More Abnormal)")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    val_score_png_path = out_dir / "normal_validation_score_distribution.png"
    plt.savefig(val_score_png_path, dpi=150)
    plt.close()

    # Calculate validation statistics
    val_std = float(np.std(val_anomaly_scores))
    val_p01 = float(np.percentile(val_anomaly_scores, 1))
    val_p05 = float(np.percentile(val_anomaly_scores, 5))
    val_min = float(np.min(val_anomaly_scores))
    val_max = float(np.max(val_anomaly_scores))
    pct_outliers_default = float(np.mean(val_preds == -1) * 100.0)

    # --------------------------------------------------------------------------
    # PART H: Threshold Calibration (From Normal Validation Percentiles)
    # --------------------------------------------------------------------------
    print("\n[PART H] Calibrating Candidate Anomaly Thresholds from Normal Percentiles...")
    p90_t = float(np.percentile(val_anomaly_scores, 90))
    p95_t = float(np.percentile(val_anomaly_scores, 95))
    p975_t = float(np.percentile(val_anomaly_scores, 97.5))
    p99_t = float(np.percentile(val_anomaly_scores, 99))
    p995_t = float(np.percentile(val_anomaly_scores, 99.5))

    threshold_candidates = [
        {
            "percentile": "90.0%",
            "threshold_score": f"{p90_t:.6f}",
            "expected_normal_false_positive_rate": "10.0%",
            "validation_observations_above": int(np.sum(val_anomaly_scores >= p90_t)),
            "validation_percentage_above": f"{np.mean(val_anomaly_scores >= p90_t) * 100.0:.2f}%",
        },
        {
            "percentile": "95.0%",
            "threshold_score": f"{p95_t:.6f}",
            "expected_normal_false_positive_rate": "5.0%",
            "validation_observations_above": int(np.sum(val_anomaly_scores >= p95_t)),
            "validation_percentage_above": f"{np.mean(val_anomaly_scores >= p95_t) * 100.0:.2f}%",
        },
        {
            "percentile": "97.5%",
            "threshold_score": f"{p975_t:.6f}",
            "expected_normal_false_positive_rate": "2.5%",
            "validation_observations_above": int(np.sum(val_anomaly_scores >= p975_t)),
            "validation_percentage_above": f"{np.mean(val_anomaly_scores >= p975_t) * 100.0:.2f}%",
        },
        {
            "percentile": "99.0%",
            "threshold_score": f"{p99_t:.6f}",
            "expected_normal_false_positive_rate": "1.0%",
            "validation_observations_above": int(np.sum(val_anomaly_scores >= p99_t)),
            "validation_percentage_above": f"{np.mean(val_anomaly_scores >= p99_t) * 100.0:.2f}%",
        },
        {
            "percentile": "99.5%",
            "threshold_score": f"{p995_t:.6f}",
            "expected_normal_false_positive_rate": "0.5%",
            "validation_observations_above": int(np.sum(val_anomaly_scores >= p995_t)),
            "validation_percentage_above": f"{np.mean(val_anomaly_scores >= p995_t) * 100.0:.2f}%",
        },
    ]

    thresholds_csv_path = out_dir / "threshold_candidates.csv"
    with open(thresholds_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fields = ["percentile", "threshold_score", "expected_normal_false_positive_rate", "validation_observations_above", "validation_percentage_above"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(threshold_candidates)
    print(f"  Saved threshold candidates to: {thresholds_csv_path}")

    # --------------------------------------------------------------------------
    # PART I: Temporal Smoothing Preparation Config
    # --------------------------------------------------------------------------
    smoothing_config = {
        "available_methods": [
            {
                "method": "moving_average",
                "description": "Applies a uniform moving average filter over a sliding window of frames.",
                "parameter": "window_size (default: 5 frames = 0.20s at 25 FPS)",
            },
            {
                "method": "median_filter",
                "description": "Applies a 1D median filter to eliminate isolated single-frame detector spikes while preserving edge transitions.",
                "parameter": "window_size (default: 5 frames = 0.20s at 25 FPS)",
            },
            {
                "method": "consecutive_persistence_rule",
                "description": "Requires anomaly score to exceed the calibrated threshold for at least K consecutive frames before triggering an alarm.",
                "parameter": "min_consecutive_frames (default: 3 frames = 0.12s at 25 FPS)",
            },
        ],
        "default_recommendation": {
            "method": "median_filter",
            "window_size": 5,
            "persistence_frames": 3,
        },
    }
    with open(out_dir / "temporal_smoothing_config.json", mode="w", encoding="utf-8") as f:
        json.dump(smoothing_config, f, indent=2)

    # --------------------------------------------------------------------------
    # PART J: Baseline Metadata JSON
    # --------------------------------------------------------------------------
    baseline_metadata = {
        "experiment_name": "Handcrafted-Feature Isolation Forest Anomaly Baseline",
        "created_at": datetime.now().isoformat(),
        "environment": {
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "numpy_version": np.__version__,
        },
        "dataset_split": split_meta,
        "selected_features": feat_spec["selected_features"],
        "preprocessing": prep_config,
        "isolation_forest_hyperparameters": {
            "n_estimators": n_estimators,
            "max_samples": str(max_samples),
            "contamination": str(contamination),
            "random_state": random_state,
        },
        "validation_score_statistics": {
            "mean": mean_s,
            "median": med_s,
            "std": val_std,
            "min": val_min,
            "p01": val_p01,
            "p05": val_p05,
            "p95": p95_s,
            "p99": p99_s,
            "max": val_max,
            "default_sklearn_outlier_percentage": pct_outliers_default,
        },
        "threshold_candidates": threshold_candidates,
        "artifacts": {
            "model_joblib": str(model_artifact_path),
            "feature_audit_csv": str(out_dir / "feature_audit.csv"),
            "extreme_observations_csv": str(out_dir / "extreme_observations.csv"),
            "selected_features_json": str(out_dir / "selected_features.json"),
            "split_metadata_json": str(out_dir / "split_metadata.json"),
            "preprocessing_json": str(out_dir / "preprocessing.json"),
            "normal_validation_scores_csv": str(val_scores_csv_path),
            "threshold_candidates_csv": str(thresholds_csv_path),
            "temporal_smoothing_config_json": str(out_dir / "temporal_smoothing_config.json"),
        },
    }

    metadata_path = out_dir / "baseline_metadata.json"
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(baseline_metadata, f, indent=2)

    # --------------------------------------------------------------------------
    # PART L: Standardized Summary Output
    # --------------------------------------------------------------------------
    pct_95 = float(threshold_candidates[1]["validation_percentage_above"].replace("%", ""))
    pct_975 = float(threshold_candidates[2]["validation_percentage_above"].replace("%", ""))
    pct_99 = float(threshold_candidates[3]["validation_percentage_above"].replace("%", ""))

    print("\n" + "=" * 52)
    print("HANDCRAFTED ANOMALY BASELINE")
    print("=" * 52)
    print(f"Training videos:             {len(split_meta['training_videos'])} ({', '.join(split_meta['training_videos'])})")
    print(f"Validation videos:           {len(split_meta['validation_videos'])} ({', '.join(split_meta['validation_videos'])})")
    print("")
    print(f"Training feature rows:       {len(train_rows)}")
    print(f"Validation feature rows:     {len(val_rows)}")
    print("")
    print(f"Selected features:           {len(selected_feature_names)} features")
    for fn in selected_feature_names:
        print(f"  - {fn}")
    print("")
    print("Isolation Forest parameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_samples:  {max_samples}")
    print(f"  contamination:{contamination}")
    print(f"  random_state: {random_state}")
    print("")
    print("Validation anomaly-score statistics:")
    print(f"  mean:   {mean_s:.6f}")
    print(f"  median: {med_s:.6f}")
    print(f"  std:    {val_std:.6f}")
    print(f"  min:    {val_min:.6f}")
    print(f"  p01:    {val_p01:.6f}")
    print(f"  p05:    {val_p05:.6f}")
    print(f"  p95:    {p95_s:.6f}")
    print(f"  p99:    {p99_s:.6f}")
    print(f"  max:    {val_max:.6f}")
    print("")
    print(f"95% threshold:               {p95_t:.6f}")
    print(f"97.5% threshold:             {p975_t:.6f}")
    print(f"99% threshold:               {p99_t:.6f}")
    print("")
    print("Percentage of validation observations above each threshold:")
    print(f"  Above 95% threshold:   {pct_95:.2f}% (Expected: 5.00%)")
    print(f"  Above 97.5% threshold: {pct_975:.2f}% (Expected: 2.50%)")
    print(f"  Above 99% threshold:   {pct_99:.2f}% (Expected: 1.00%)")
    print("")
    print(f"Model saved to:              {model_artifact_path}")
    print(f"Preprocessing saved to:      {out_dir / 'preprocessing.json'}")
    print(f"Metadata saved to:           {metadata_path}")
    print("=" * 52)

    return baseline_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build handcrafted feature Isolation Forest anomaly baseline."
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default="outputs/training_dataset/normal_cluster_features.csv",
        help="Path to merged normal cluster features CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/anomaly_baseline",
        help="Output directory for anomaly baseline artifacts.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of Isolation Forest trees (default: 200).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    build_handcrafted_anomaly_baseline(
        merged_features_csv=args.features_csv,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
