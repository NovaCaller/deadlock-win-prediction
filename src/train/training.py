import logging

# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
from tqdm import tqdm, trange

from src.train.early_stopping import EarlyStopping
from src.train.util import test_loop


def training(model, train_loader, val_loader, loss_function, optimizer, number_of_epochs, early_stopping_patience: int = 5, early_stopping_min_improvement_pct: float = 0.01, use_tqdm: bool = True) -> tuple[list[float], list[float], list[float], list[float]]:
    early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_improvement_pct)

    training_losses: list[float] = []
    training_accuracies: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    best_epoch = number_of_epochs
    epoch_iter = trange(1, number_of_epochs + 1, desc="Epochs") if use_tqdm else range(1, number_of_epochs + 1)
    for epoch in epoch_iter:
        train_loss, train_acc = _train_loop(model, train_loader, loss_function, optimizer)
        val_loss, val_acc = test_loop(model, val_loader, loss_function)

        training_losses.append(train_loss)
        training_accuracies.append(train_acc)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        logging.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Validation Loss={val_loss:.4f}, Validation Acc={val_acc:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write(f"early stopping at epoch {epoch:02d}")
            best_epoch = epoch - early_stopping_patience
            logging.info(f"Best Epoch: {best_epoch}")
            break

    early_stopping.load_best_model(model)
    return training_losses[:best_epoch], training_accuracies[:best_epoch], validation_losses[:best_epoch], validation_accuracies[:best_epoch]


def _train_loop(model, loader, loss_function, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False, dynamic_ncols=True):
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        # Compute accuracy
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc