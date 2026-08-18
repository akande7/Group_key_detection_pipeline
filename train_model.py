from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PREPARED_DATA_DIRECTORY = Path("prepared_data")
MODELS_DIRECTORY = Path("models")

FEATURE_CONFIGURATION_FILE = (
    PREPARED_DATA_DIRECTORY / "feature_configuration.json"
)
NORMAL_L2_FEATURES_FILE = (
    PREPARED_DATA_DIRECTORY / "normal_l2_features.csv"
)
NORMAL_CROSS_LAYER_FEATURES_FILE = (
    PREPARED_DATA_DIRECTORY / "normal_cross_layer_features.csv"
)

RANDOM_STATE = 42
CONTAMINATION = "auto"


def load_feature_configuration(path: Path) -> dict:
    """Load the feature selection metadata produced by prepare_data.py."""

    if not path.exists():
        raise FileNotFoundError(
            "Feature configuration not found. "
            f"Run prepare_data.py first: {path}"
        )

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_training_features(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    """Load a model-ready feature matrix and validate its columns."""

    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Training dataset contains no observations: {path}")

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            + ", ".join(missing_columns)
        )

    return dataframe[feature_columns]


def build_anomaly_detector() -> Pipeline:
    """
    Build an unsupervised anomaly detector for normal-only traffic.

    The model learns the distribution of legitimate Wi-Fi behaviour and
    flags windows that deviate from that baseline during inference.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=200,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                    max_samples=48,     
                    max_features = 0.25,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    features: pd.DataFrame,
    model_name: str,
) -> Pipeline:
    """Fit one anomaly detector on normal traffic features."""

    model = build_anomaly_detector()
    model.fit(features)

    print(f"{model_name} model trained on {len(features)} observations")
    print(f"{model_name} feature count: {features.shape[1]}")

    return model


def save_model(
    model: Pipeline,
    output_path: Path,
) -> None:
    """Persist a trained model pipeline to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def main() -> None:
    configuration = load_feature_configuration(FEATURE_CONFIGURATION_FILE)

    l2_features = load_training_features(
        NORMAL_L2_FEATURES_FILE,
        configuration["selected_l2_features"],
    )

    cross_layer_features = load_training_features(
        NORMAL_CROSS_LAYER_FEATURES_FILE,
        configuration["selected_cross_layer_features"],
    )

    l2_model = train_model(
        l2_features,
        model_name="L2",
    )

    cross_layer_model = train_model(
        cross_layer_features,
        model_name="Cross-layer",
    )

    MODELS_DIRECTORY.mkdir(exist_ok=True)

    l2_model_path = MODELS_DIRECTORY / "l2_anomaly_detector.joblib"
    cross_layer_model_path = (
        MODELS_DIRECTORY / "cross_layer_anomaly_detector.joblib"
    )

    save_model(l2_model, l2_model_path)
    save_model(cross_layer_model, cross_layer_model_path)

    training_metadata = {
        "model_type": "IsolationForest",
        "training_data": "normal_only",
        "normal_observations": len(l2_features),
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "l2_model_path": str(l2_model_path),
        "cross_layer_model_path": str(cross_layer_model_path),
        "selected_l2_features": configuration["selected_l2_features"],
        "selected_cross_layer_features": (
            configuration["selected_cross_layer_features"]
        ),
    }

    metadata_path = MODELS_DIRECTORY / "training_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            training_metadata,
            file,
            indent=4,
        )

    print("\nModel training complete")
    print("-" * 50)
    print(f"Normal observations used: {len(l2_features)}")
    print(f"L2 model saved to: {l2_model_path}")
    print(f"Cross-layer model saved to: {cross_layer_model_path}")
    print(f"Training metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
