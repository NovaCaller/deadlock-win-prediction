import logging
import pickle
from pathlib import Path

from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from src.common.set_up_logging import set_up_logging

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_PATH = Path("../../model")
DATA_PATH = MODEL_PATH / "training_dataframe.pkl"

LABEL_COLUMN = "winning_team"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOG_LEVEL = logging.INFO

console = Console()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def evaluate_model_forest(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    logging.info(f"\n===== {name} =====")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"F1-score: {f1_score(y_test, y_pred):.4f}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        logging.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)

    assert DATA_PATH.exists(), "training_dataframe.pkl not found"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    logging.info(f"Loaded dataframe with shape {df.shape}")

    assert LABEL_COLUMN in df.columns, f"Label column '{LABEL_COLUMN}' missing"

    X = df.drop(columns=[LABEL_COLUMN]).values
    y = df[LABEL_COLUMN].values.astype(np.int32)

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logging.info(
        f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}"
    )

    # --------------------------------------------------
    # Random Forest Classifier
    # --------------------------------------------------
    random_forest = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=None,    # Full depth
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,         # Utilize all available CPU cores
    )

    random_forest.fit(X_train, y_train)

    evaluate_model("Random Forest", random_forest, X_test, y_test)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    with open(MODEL_PATH / "random_forest.pkl", "wb") as f:
        pickle.dump(random_forest, f)

    logging.info("Saved random forest model to disk.")