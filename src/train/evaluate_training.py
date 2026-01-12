from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAIN_LOG_FILE_PATH = SCRIPT_DIR.parent.parent / "logs"  # Go up one level first

def plot_training_metrics(parquet_path: str, save_path: str = None):
    """
    Reads training metrics from a parquet file and plots loss/accuracy curves.

    Args:
        parquet_path: Path to the parquet file
        save_path: Optional path to save the figure (will append _loss and _acc)
    """
    df = pd.read_parquet(parquet_path)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], color='blue', label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], color='red', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_loss.png", dpi=150)
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_acc'], color='blue', label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], color='red', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_acc.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    plot_training_metrics(str(TRAIN_LOG_FILE_PATH / "train_log.parquet"))