# noinspection PyPackageRequirements
import torch
from tqdm import tqdm


def test_loop(model, loader, loss_function):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True):
            logits = model(X_batch).squeeze(1)
            loss = loss_function(logits, y_batch)
            running_loss += loss.item() * X_batch.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc