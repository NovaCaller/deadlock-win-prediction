import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
import logging


def evaluate_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred)
    }

    logging.info(f"\n===== {name} =====")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        metrics['bce'] = log_loss(y_test, y_prob)
        metrics['mse'] = mean_squared_error(y_test, y_prob)

        logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logging.info(f"BCE (Log Loss): {metrics['bce']:.4f}")
        logging.info(f"MSE: {metrics['mse']:.4f}")

    return metrics


def evaluate_classifier_pytorch(name, model, X_test, y_test, device):
    model.eval()

    # Convert DataFrame/Series to tensors
    X_test = torch.FloatTensor(X_test.values)
    y_test = torch.FloatTensor(y_test.values)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        y_prob = model(X_test)
        y_prob = torch.sigmoid(y_prob.squeeze())

        # Get predictions
        y_pred = (y_prob > 0.5).float()

        # Calculate metrics
        accuracy = (y_pred == y_test).float().mean()

        # BCE Loss
        bce_loss = torch.nn.functional.binary_cross_entropy(y_prob, y_test)

        # MSE
        mse = torch.nn.functional.mse_loss(y_prob, y_test)

        # Convert to numpy for ROC-AUC calculation
        y_prob_np = y_prob.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        roc_auc = roc_auc_score(y_test_np, y_prob_np)

    logging.info(f"\n===== {name} (PyTorch) =====")
    logging.info(f"Accuracy: {accuracy.item():.4f}")
    logging.info(f"ROC-AUC: {roc_auc:.4f}")
    logging.info(f"BCE Loss: {bce_loss.item():.4f}")
    logging.info(f"MSE: {mse.item():.4f}")

    return {
        'accuracy': accuracy.item(),
        'roc_auc': roc_auc,
        'bce': bce_loss.item(),
        'mse': mse.item()
    }


def average_metrics(all_metrics):
    """Calculate mean and std for a list of metric dictionaries"""
    if not all_metrics:
        return {}

    # Get all metric names from first run
    metric_names = all_metrics[0].keys()

    averaged = {}
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        averaged[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    return averaged


def log_averaged_metrics(name, averaged_metrics):
    """Log averaged metrics with mean ± std"""
    print(f"\n===== {name} (Averaged over {len(list(averaged_metrics.values())[0]['values'])} runs) =====")

    for metric_name, stats in averaged_metrics.items():
        metric_display = metric_name.replace('_', ' ').upper()
        if metric_name == 'bce':
            metric_display = 'BCE (Log Loss)'
        print(f"{metric_display}: {stats['mean']:.4f} ± {stats['std']:.4f}")