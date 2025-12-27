import logging

# noinspection PyPackageRequirements
import torch
from tqdm import tqdm, trange

from src.train.util import test_loop


def training(model, train_loader, val_loader, loss_function, optimizer, number_of_epochs):
    for epoch in trange(1, number_of_epochs + 1, desc="Epochs"):
        train_loss, train_acc = _train_loop(model, train_loader, loss_function, optimizer)
        val_loss, val_acc = test_loop(model, val_loader, loss_function)

        tqdm.write(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Validation Loss={val_loss:.4f}, Validation Acc={val_acc:.4f}"
        )


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