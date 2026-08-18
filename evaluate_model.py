"""
Evaluate all three detection approaches (rule-based baseline, L2-only
Isolation Forest, cross-layer Isolation Forest) against:

  - a held-out normal session the models never trained on (false-positive
    rate check), and
  - the attack capture (detection / true-positive rate check).

This script assumes prepare_data.py and train_model.py have already been
run, producing prepared_data/feature_configuration.json and the two
trained model files in models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PREPARED_DATA_DIRECTORY = Path("prepared_data")
MODELS_DIRECTORY = Path("models")

FEATURE_CONFIGURATION_FILE = (
    PREPARED_DATA_DIRECTORY / "feature_configuration.json"
)
L2_MODEL_FILE = MODELS_DIRECTORY / "l2_anomaly_detector.joblib"
CROSS_LAYER_MODEL_FILE = (
    MODELS_DIRECTORY / "cross_layer_anomaly_detector.joblib"
)

# Held-out normal session (never used in training) -- raw extract_features.py
# output, i.e. NOT yet passed through prepare_data.py.
VALIDATION_NORMAL_FILE = Path("testing_normal_features.csv")

# Attack capture -- raw extract_features.py output.
ATTACK_FILE = Path("attack_features.csv")


def load_feature_configuration(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature configuration not found. Run prepare_data.py first: {path}"
        )
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found. Run train_model.py first: {path}"
        )
    return joblib.load(path)


def clean_numeric_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Mirror the cleaning applied in prepare_data.py so evaluation data
    is treated consistently with training data."""
    cleaned = dataframe.copy()
    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[feature_columns] = cleaned[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    cleaned[feature_columns] = cleaned[feature_columns].fillna(0)
    return cleaned


def load_raw_dataset(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"Dataset contains no observations: {path}")
    missing = [c for c in required_columns if c not in dataframe.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return dataframe


def rule_based_predict(dataframe: pd.DataFrame) -> pd.Series:
    """
    Simple rule-based baseline: flag any window containing at least one
    cross-layer inconsistency (broadcast MAC destination paired with a
    unicast IP destination), the signature associated with GTK abuse.

    Returns a boolean Series: True = flagged as anomalous.
    """
    return dataframe["cross_layer_inconsistency_count"] > 0


def isolation_forest_predict(model, dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """
    Returns a boolean Series: True = flagged as anomalous.
    scikit-learn's IsolationForest.predict() returns 1 for normal, -1
    for anomaly.
    """
    predictions = model.predict(dataframe[feature_columns])
    return pd.Series(predictions == -1, index=dataframe.index)


def summarise(
    name: str,
    flagged: pd.Series,
    window_numbers: pd.Series,
    known_attack_windows: set[int] | None = None,
) -> dict:
    total = len(flagged)
    flagged_count = int(flagged.sum())
    rate = flagged_count / total if total else 0.0

    flagged_window_numbers = sorted(
        int(w) for w in window_numbers[flagged.values]
    )

    result = {
        "approach": name,
        "total_windows": total,
        "flagged_windows": flagged_count,
        "flagged_rate": round(rate, 4),
        "flagged_window_numbers": flagged_window_numbers,
    }

    if known_attack_windows is not None:
        flagged_set = set(flagged_window_numbers)
        true_positives = flagged_set & known_attack_windows
        false_negatives = known_attack_windows - flagged_set
        false_positives = flagged_set - known_attack_windows
        true_negatives = (
            set(int(w) for w in window_numbers) - flagged_set - known_attack_windows
        )

        tp, fp, fn, tn = (
            len(true_positives),
            len(false_positives),
            len(false_negatives),
            len(true_negatives),
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = (tp + tn) / total if total else 0.0

        result["true_positive_windows"] = sorted(true_positives)
        result["false_negative_windows"] = sorted(false_negatives)
        result["false_positive_windows_within_attack_capture"] = sorted(
            false_positives
        )
        result["precision"] = round(precision, 4)
        result["recall"] = round(recall, 4)
        result["f1_score"] = round(f1_score, 4)
        result["accuracy"] = round(accuracy, 4)

    return result


def main() -> None:
    configuration = load_feature_configuration(FEATURE_CONFIGURATION_FILE)
    l2_features = configuration["selected_l2_features"]
    cross_layer_features = configuration["selected_cross_layer_features"]

    l2_model = load_model(L2_MODEL_FILE)
    cross_layer_model = load_model(CROSS_LAYER_MODEL_FILE)

    all_required_columns = sorted(
        set(l2_features) | set(cross_layer_features) | {"cross_layer_inconsistency_count"}
    )

    # ---- Held-out normal (false-positive check) ----
    validation = load_raw_dataset(VALIDATION_NORMAL_FILE, all_required_columns)
    validation = clean_numeric_features(validation, all_required_columns)

    validation_results = [
        summarise(
            "Rule-based (normal/held-out)",
            rule_based_predict(validation),
            validation["window_number"],
        ),
        summarise(
            "L2-only Isolation Forest (normal/held-out)",
            isolation_forest_predict(l2_model, validation, l2_features),
            validation["window_number"],
        ),
        summarise(
            "Cross-layer Isolation Forest (normal/held-out)",
            isolation_forest_predict(cross_layer_model, validation, cross_layer_features),
            validation["window_number"],
        ),
    ]

    # ---- Attack data (true-positive / detection check) ----
    attack = load_raw_dataset(ATTACK_FILE, all_required_columns)
    attack = clean_numeric_features(attack, all_required_columns)

    # Ground truth: windows where the injected attack signature is
    # actually present, derived directly from the raw cross-layer
    # inconsistency count rather than hard-coded, so this stays correct
    # if the attack capture is regenerated.
    known_attack_windows = set(
        int(w)
        for w in attack.loc[
            attack["cross_layer_inconsistency_count"] > 0, "window_number"
        ]
    )

    attack_results = [
        summarise(
            "Rule-based (attack)",
            rule_based_predict(attack),
            attack["window_number"],
            known_attack_windows,
        ),
        summarise(
            "L2-only Isolation Forest (attack)",
            isolation_forest_predict(l2_model, attack, l2_features),
            attack["window_number"],
            known_attack_windows,
        ),
        summarise(
            "Cross-layer Isolation Forest (attack)",
            isolation_forest_predict(cross_layer_model, attack, cross_layer_features),
            attack["window_number"],
            known_attack_windows,
        ),
    ]

    results = pd.DataFrame(validation_results + attack_results)

    print("\nEvaluation complete")
    print("-" * 70)
    print(f"Known attack windows (ground truth): {sorted(known_attack_windows)}")
    print()
    for row in validation_results + attack_results:
        print(f"{row['approach']}")
        print(f"  Flagged: {row['flagged_windows']}/{row['total_windows']} "
              f"({row['flagged_rate']*100:.1f}%)")
        print(f"  Flagged window numbers: {row['flagged_window_numbers']}")
        if "true_positive_windows" in row:
            print(f"  True positives (correctly caught attack windows): {row['true_positive_windows']}")
            print(f"  False negatives (missed attack windows): {row['false_negative_windows']}")
            print(f"  False positives (flagged but not actually attack): {row['false_positive_windows_within_attack_capture']}")
            print(f"  Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | "
                  f"F1-score: {row['f1_score']:.4f} | Accuracy: {row['accuracy']:.4f}")
        print()

    results.to_csv("evaluation_results.csv", index=False)
    print("Results saved to: evaluation_results.csv")


if __name__ == "__main__":
    main()