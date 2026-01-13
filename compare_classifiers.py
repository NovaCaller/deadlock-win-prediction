import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# early torch setup
from src.common.pytorch_setup import ensure_torch
from src.train.dataloaders import get_dataloaders

ensure_torch()

# early config setup
from src.common.load_config import load_model_config
MODEL_PATH: Path = Path("model")
assert MODEL_PATH.exists()
assert (MODEL_PATH / "model.toml").exists()
MODEL_CONFIG: dict = load_model_config(MODEL_PATH / "model.toml")
print(f"Loaded model config: {MODEL_CONFIG}")

# early reproducibility setup
from src.common.reproducibility import ensure_reproducibility
ensure_reproducibility(MODEL_CONFIG["seed"])

# continue normally with imports / global vars
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch

from src.common.predictors import get_new_fully_connected_model
from src.common.set_up_logging import set_up_logging
from src.other_classifiers.eval import evaluate_classifier, evaluate_classifier_pytorch, average_metrics, \
    log_averaged_metrics
from src.train.training import training
from train_model import OPTIMIZER_TYPE, LEARNING_RATE, LOSS_FUNCTION, NUMBER_OF_EPOCHS, BATCH_SIZE, \
    VALIDATION_PERCENTAGE, TEST_PERCENTAGE

# ---------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------

DATA_PATH = MODEL_PATH / "training_dataframe.parquet"
NEURAL_NET_WEIGHTS = MODEL_PATH / "model_weights.pth"
NEURAL_NET_CONFIG = MODEL_PATH / "model.toml"

LABEL_COLUMN = "winning_team"
TEST_SIZE = 0.15
N_RUNS = 25
LOG_LEVEL = logging.INFO




if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    assert DATA_PATH.exists(), "training_dataframe not found"
    assert NEURAL_NET_WEIGHTS.exists(), "weights not found"

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------
    df = pd.read_parquet(DATA_PATH)

    logging.info(f"loaded data: {df.shape} rows")
    assert LABEL_COLUMN in df.columns, f"label column '{LABEL_COLUMN}' not in dataframe."

    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN].astype(np.int32)

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # Dictionaries to store metrics across runs
    all_metrics = {
        'Dummy (Most Frequent)': [],
        'Dummy (Random)': [],
        'Logistic Regression': [],
        'Random Forest': [],
        'Neural Network': []
    }

    # --------------------------------------------------
    # Run evaluation N_RUNS times with different splits
    # --------------------------------------------------
    for run in range(N_RUNS):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"RUN {run + 1}/{N_RUNS}")
        logging.info(f"{'=' * 60}")

        # Use different random state for each split
        random_state = MODEL_CONFIG["seed"] + run

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=random_state,
            stratify=y,
        )

        logging.info(f"Data Split: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        # --------------------------------------------------
        # Dummy-Classifiers
        # --------------------------------------------------
        dummy_most_frequent = DummyClassifier(strategy="most_frequent")
        dummy_random = DummyClassifier(strategy="uniform", random_state=random_state)

        dummy_most_frequent.fit(X_train, y_train)
        dummy_random.fit(X_train, y_train)

        metrics = evaluate_classifier("Dummy (Most Frequent)", dummy_most_frequent, X_test, y_test)
        all_metrics['Dummy (Most Frequent)'].append(metrics)

        metrics = evaluate_classifier("Dummy (Random)", dummy_random, X_test, y_test)
        all_metrics['Dummy (Random)'].append(metrics)

        # --------------------------------------------------
        # Logistic Regression
        # --------------------------------------------------
        log_reg = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        log_reg.fit(X_train, y_train)
        metrics = evaluate_classifier("Logistic Regression", log_reg, X_test, y_test)
        all_metrics['Logistic Regression'].append(metrics)

        # --------------------------------------------------
        # Random Forest Classifier
        # --------------------------------------------------
        random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        random_forest.fit(X_train, y_train)
        metrics = evaluate_classifier("Random Forest", random_forest, X_test, y_test)
        all_metrics['Random Forest'].append(metrics)

        # --------------------------------------------------
        # train NN
        # --------------------------------------------------

        # load model
        model = get_new_fully_connected_model(MODEL_CONFIG["number_of_hidden_layers"], MODEL_CONFIG["number_of_features"],
                                              MODEL_CONFIG["neurons_per_layer"])
        model = model.to(device)
        train_df = X_train.copy()
        train_df[LABEL_COLUMN] = y_train
        tensor = torch.from_numpy(train_df.values)

        # load data
        train_loader, val_loader, _, number_of_features = get_dataloaders(tensor, BATCH_SIZE,
                                                                                    VALIDATION_PERCENTAGE / (1 - (VALIDATION_PERCENTAGE + TEST_PERCENTAGE)),
                                                                                    0, device,
                                                                                    random_state)
        assert number_of_features == MODEL_CONFIG["number_of_features"]

        # train model
        optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LEARNING_RATE)
        training(model, train_loader, val_loader, LOSS_FUNCTION, optimizer, NUMBER_OF_EPOCHS, use_tqdm=False)

        metrics = evaluate_classifier_pytorch(
            "Neural Network",
            model,
            X_test,
            y_test,
            device
        )
        all_metrics['Neural Network'].append(metrics)

    # --------------------------------------------------
    # Calculate and print averaged metrics
    # --------------------------------------------------
    print(f"\n\n{'='*60}")
    print(f"AVERAGED RESULTS ACROSS {N_RUNS} RUNS")
    print(f"{'='*60}")

    for model_name, metrics_list in all_metrics.items():
        averaged = average_metrics(metrics_list)
        log_averaged_metrics(model_name, averaged)