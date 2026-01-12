import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

from src.common.set_up_logging import set_up_logging

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_PATH = Path("../../model")
DATA_PATH = MODEL_PATH / "training_dataframe.pkl"

LABEL_COLUMN = "winning_team"  # ggf. anpassen
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOG_LEVEL = logging.INFO


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def evaluate_model_regression(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    logging.info(f"\n===== {name} =====")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    f1 = f1_score(y_test, y_pred, zero_division=0)
    logging.info(f"F1-score: {f1:.4f}")

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

    # Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # Dummy classifiers
    # --------------------------------------------------
    dummy_most_frequent = DummyClassifier(strategy="most_frequent")
    dummy_random = DummyClassifier(strategy="uniform", random_state=RANDOM_STATE)

    dummy_most_frequent.fit(X_train_scaled, y_train)
    dummy_random.fit(X_train_scaled, y_train)

    evaluate_model("Dummy (most frequent)", dummy_most_frequent, X_test_scaled, y_test)
    evaluate_model("Dummy (random)", dummy_random, X_test_scaled, y_test)

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------
    log_reg = LogisticRegression(
        max_iter=2000,  # Höherer Wert zum Vermeiden der ConvergenceWarning
        class_weight="balanced",
        solver="lbfgs",  # Alternativen: "sag", "saga" für große Datensätze
    )

    log_reg.fit(X_train_scaled, y_train)

    evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    with open(MODEL_PATH / "logistic_regression.pkl", "wb") as f:
        pickle.dump(log_reg, f)

    logging.info("Saved logistic regression model to disk.")