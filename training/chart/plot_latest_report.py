from pathlib import Path
from typing import Any

from .globals import discover_reports, load_report


def get_history(report: dict[str, Any]) -> dict[str, list[float]]:
    history = report.get("history", {})
    return {key: [float(value) for value in values] for key, values in history.items()}

def latest_report() -> Path | None:
    reports = discover_reports()
    return reports[-1] if reports else None

def parse_index(report_path: Path) -> int | None:
    return report_path.stem.split("_")[-1] if report_path.stem.startswith("training_report_") else None

def main() -> Path | None:
    report_path = latest_report()
    if report_path is None:
        return None

    index = parse_index(report_path)
    report = load_report(report_path)
    history = get_history(report)
    if not history:
        return report_path

    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    fig.suptitle(f"Latest training report: {str(index) if index is not None else 'unknown'}")

    axes[0].plot(epochs, history.get("loss", []), label="train loss", linewidth=2)
    if history.get("val_loss"):
        axes[0].plot(epochs, history.get("val_loss", []), label="val loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("accuracy", []), label="train accuracy", linewidth=2)
    if history.get("val_accuracy"):
        axes[1].plot(epochs, history.get("val_accuracy", []), label="val accuracy", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, history.get("recall", []), label="train recall", linewidth=2)
    if history.get("val_recall"):
        axes[2].plot(epochs, history.get("val_recall", []), label="val recall", linewidth=2)
    axes[2].set_title("Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recall")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    print(f"Showing latest training curves from {report_path}...")
    plt.show()
    return report_path

if __name__ == "__main__":
    raise SystemExit(main())