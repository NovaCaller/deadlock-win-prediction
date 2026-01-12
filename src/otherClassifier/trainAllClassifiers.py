import logging
import pickle
from pathlib import Path
import json

# ML / Statistik
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
from src.common.predictors import load_fully_connected_model
from src.common.set_up_logging import set_up_logging
from src.otherClassifier.logistic_regression import evaluate_model_regression
from src.otherClassifier.random_forest import evaluate_model_forest
from src.prep.util import normalize_df

# ---------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / ("model")
DATA_PATH = MODEL_PATH / "training_dataframe.pkl"
NEURAL_NET_WEIGHTS = MODEL_PATH / "model_weights.pth"
NEURAL_NET_CONFIG = MODEL_PATH / "model.toml"

LABEL_COLUMN = "winning_team"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOG_LEVEL = logging.INFO

# ---------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------

if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)

    assert MODEL_PATH.exists(), "model-Ordner nicht gefunden"
    assert DATA_PATH.exists(), "training_dataframe.pkl nicht gefunden"
    assert NEURAL_NET_WEIGHTS.exists(), "Vortrainierte Gewichte nicht gefunden."

    # --------------------------------------------------
    # Laden der Daten
    # --------------------------------------------------
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    logging.info(f"Daten geladen: {df.shape} Zeilen")
    assert LABEL_COLUMN in df.columns, f"Labelspalte '{LABEL_COLUMN}' nicht in den Daten enthalten"

    X = df.drop(columns=[LABEL_COLUMN]).values
    y = df[LABEL_COLUMN].values.astype(np.int32)

    # --------------------------------------------------
    # Train-Test-Split erstellen
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logging.info(f"Daten Split: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # Dummy-Klassifikatoren
    # --------------------------------------------------
    dummy_most_frequent = DummyClassifier(strategy="most_frequent")
    dummy_random = DummyClassifier(strategy="uniform", random_state=RANDOM_STATE)

    dummy_most_frequent.fit(X_train_scaled, y_train)
    dummy_random.fit(X_train_scaled, y_train)

    evaluate_model_regression("Dummy (Most Frequent)", dummy_most_frequent, X_test_scaled, y_test)
    evaluate_model_regression("Dummy (Random)", dummy_random, X_test_scaled, y_test)

    # --------------------------------------------------
    # Logistische Regression
    # --------------------------------------------------
    log_reg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    log_reg.fit(X_train_scaled, y_train)
    evaluate_model_regression("Logistische Regression", log_reg, X_test_scaled, y_test)

    # --------------------------------------------------
    # Random Forest Classifier
    # --------------------------------------------------
    random_forest = RandomForestClassifier(
        n_estimators=100,  # Anzahl der Bäume
        max_depth=None,    # Unbeschränkte Tiefe
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,         # Alle verfügbaren CPU-Kerne verwenden
    )
    random_forest.fit(X_train, y_train)
    evaluate_model_forest("Random Forest", random_forest, X_test, y_test)

    # --------------------------------------------------
    # Vortrainiertes neuronales Netzwerk laden
    # --------------------------------------------------


    from src.common.load_config import load_model_config
    MODEL_CONFIG = load_model_config(NEURAL_NET_CONFIG)
    normalization_params_path = MODEL_PATH / "normalization_params.json"
    assert normalization_params_path.exists()
    weights_path = MODEL_PATH / "model_weights.pth"
    assert weights_path.exists()

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load model with weights
    model = load_fully_connected_model(weights_path, MODEL_CONFIG["number_of_hidden_layers"],
                                       MODEL_CONFIG["number_of_features"], MODEL_CONFIG["neurons_per_layer"])
    model = model.to(device)

    # load normalization parameters
    with open(normalization_params_path) as f:
        loaded_params: dict[str, list[float]] = json.load(f)
    normalization_params: dict[str, tuple[float, float]] = {
        k: (v[0], v[1])
        for k, v in loaded_params.items()
    }

    df = normalize_df(df, ["timestamp"], normalization_params)
    logging.info("done normalizing timestamps.")
    logging.info("finalized dataframe.")
    logging.info("wrote normalization parameters to disk.")
    tensor = torch.from_numpy(df.values)
    logging.info(
        f"converted to tensor with {tensor.shape[0]} rows and {tensor.shape[1]} columns.")
    torch.save(tensor, MODEL_PATH / "training_tensor.pt")
    logging.info("wrote tensor to disk.")

    # Modell evaluieren
    from src.train.util import predict_and_evaluate

    predict_and_evaluate("Neuronales Netzwerk", model, X_test_scaled, y_test, device)