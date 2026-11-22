from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset from CSV, or fetch and save it if missing."""
    if data_path.exists():
        return pd.read_csv(data_path)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_california_housing(as_frame=True).frame
    df.to_csv(data_path, index=False)
    return df


def build_pipeline() -> Pipeline:
    """Create the winning preprocessing + Random Forest pipeline."""
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(y_true, y_pred) -> dict:
    """Compute regression evaluation metrics."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main() -> None:
    print("Loading data...")
    df = load_data(DATA_PATH)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print(f"Evaluation Metrics: {metrics}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"Model saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
