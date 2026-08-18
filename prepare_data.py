from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


NORMAL_FILE = Path("combined_normal_features.csv")
ATTACK_FILE = Path("attack_features.csv")

OUTPUT_DIRECTORY = Path("prepared_data")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)


# Columns used only to identify the observation window.
METADATA_COLUMNS = [
    "window_number",
    "window_start_seconds",
    "window_end_seconds",
]


# Layer 2 and traffic-behaviour features.
L2_FEATURES = [
    "dot11_frame_count",
    "management_frame_count",
    "control_frame_count",
    "data_frame_count",
    "beacon_count",
    "broadcast_mac_count",
    "multicast_mac_count",
    "unicast_mac_count",
    "eapol_count",
    "protected_frame_count",
    "total_packet_bytes",
    "average_packet_size",
    "packet_rate_per_second",
    "broadcast_mac_ratio",
    "unicast_mac_ratio",
]


# Layer 3 and cross-layer features added to the cross-layer model.
CROSS_LAYER_FEATURES = [
    "ip_count",
    "arp_count",
    "broadcast_mac_broadcast_ip_count",
    "broadcast_mac_multicast_ip_count",
    "cross_layer_inconsistency_count",
    "cross_layer_inconsistency_ratio",
]


# These features must be retained even if they are constant in normal traffic.
# Their normal value may legitimately be zero, but they may become non-zero
# during GTK abuse.
MANDATORY_FEATURES = [
    "arp_count",
    "cross_layer_inconsistency_count",
    "cross_layer_inconsistency_ratio",
]


def load_dataset(
    path: Path,
    label: int,
    traffic_type: str,
) -> pd.DataFrame:
    """Load one extracted feature dataset and attach evaluation metadata."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset contains no observations: {path}")

    dataframe["label"] = label
    dataframe["traffic_type"] = traffic_type
    dataframe["source_file"] = path.name

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Ensure that the extracted dataset contains all required features."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing_columns)
        )


def clean_numeric_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert feature columns to numeric values and handle invalid values.
    """

    cleaned = dataframe.copy()

    for column in feature_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Replace positive and negative infinity with missing values.
    cleaned[feature_columns] = cleaned[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Missing counts and ratios are replaced with zero.
    cleaned[feature_columns] = cleaned[feature_columns].fillna(0)

    return cleaned


def identify_constant_features(
    normal_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """
    Identify constant features using only the normal training dataset.

    Mandatory security features are retained even when their value is
    constant during normal traffic.
    """

    return [
        column
        for column in feature_columns
        if normal_dataframe[column].nunique(dropna=False) <= 1
        and column not in MANDATORY_FEATURES
    ]


def save_feature_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Save only the selected model-input features."""

    dataframe[feature_columns].to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    all_features = L2_FEATURES + CROSS_LAYER_FEATURES

    # Load and prepare the normal dataset.
    normal = load_dataset(
        NORMAL_FILE,
        label=0,
        traffic_type="normal",
    )

    validate_required_columns(
        normal,
        all_features,
        "Normal dataset",
    )

    normal = clean_numeric_features(
        normal,
        all_features,
    )

    # Identify features that provide no variation in normal traffic.
    constant_features = identify_constant_features(
        normal,
        all_features,
    )

    selected_l2_features = [
        feature
        for feature in L2_FEATURES
        if feature not in constant_features
    ]

    selected_cross_layer_features = [
        feature
        for feature in all_features
        if feature not in constant_features
    ]

    # Save the complete prepared normal dataset, including metadata.
    normal.to_csv(
        OUTPUT_DIRECTORY / "prepared_normal.csv",
        index=False,
    )

    # Save model-ready feature matrices.
    save_feature_matrix(
        normal,
        selected_l2_features,
        OUTPUT_DIRECTORY / "normal_l2_features.csv",
    )

    save_feature_matrix(
        normal,
        selected_cross_layer_features,
        OUTPUT_DIRECTORY / "normal_cross_layer_features.csv",
    )

    attack_loaded = False
    attack_observations = 0

    # Load the attack dataset only when it exists.
    if ATTACK_FILE.exists():
        attack = load_dataset(
            ATTACK_FILE,
            label=1,
            traffic_type="gtk_abuse",
        )

        validate_required_columns(
            attack,
            all_features,
            "Attack dataset",
        )

        attack = clean_numeric_features(
            attack,
            all_features,
        )

        attack_observations = len(attack)

        attack.to_csv(
            OUTPUT_DIRECTORY / "prepared_attack.csv",
            index=False,
        )

        save_feature_matrix(
            attack,
            selected_l2_features,
            OUTPUT_DIRECTORY / "attack_l2_features.csv",
        )

        save_feature_matrix(
            attack,
            selected_cross_layer_features,
            OUTPUT_DIRECTORY / "attack_cross_layer_features.csv",
        )

        combined = pd.concat(
            [normal, attack],
            ignore_index=True,
        )

        combined.to_csv(
            OUTPUT_DIRECTORY / "combined_dataset.csv",
            index=False,
        )

        attack_loaded = True

    configuration = {
        "metadata_columns": METADATA_COLUMNS,
        "original_l2_features": L2_FEATURES,
        "original_cross_layer_features": CROSS_LAYER_FEATURES,
        "mandatory_features_retained": MANDATORY_FEATURES,
        "constant_features_removed": constant_features,
        "selected_l2_features": selected_l2_features,
        "selected_cross_layer_features": selected_cross_layer_features,
        "normal_observations": len(normal),
        "attack_observations": attack_observations,
        "attack_dataset_loaded": attack_loaded,
    }

    with open(
        OUTPUT_DIRECTORY / "feature_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )

    print("\nData preparation complete")
    print("-" * 50)
    print(f"Normal observations: {len(normal)}")
    print(f"Attack observations: {attack_observations}")
    print(f"Attack dataset loaded: {attack_loaded}")
    print(f"Constant features removed: {constant_features}")
    print(
        "Mandatory features retained: "
        f"{MANDATORY_FEATURES}"
    )
    print(
        f"L2 feature count: {len(selected_l2_features)}"
    )
    print(
        "Cross-layer feature count: "
        f"{len(selected_cross_layer_features)}"
    )
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()