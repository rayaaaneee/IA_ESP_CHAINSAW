from .globals import REPORT_DIR, discover_reports, load_report, report_index


def final_metric_series() -> tuple[list[int], list[float | None], list[float | None], list[float | None], list[float | None]]:
    report_numbers: list[int] = []
    val_accuracy: list[float | None] = []
    val_recall: list[float | None] = []
    test_accuracy: list[float | None] = []
    test_recall: list[float | None] = []

    for report_path in discover_reports():
        report = load_report(report_path)
        validation_metrics = report.get("validation_metrics", {})
        test_metrics = report.get("test_metrics", {})

        report_numbers.append(report_index(report_path))
        val_accuracy.append(validation_metrics.get("accuracy"))
        val_recall.append(validation_metrics.get("recall"))
        test_accuracy.append(test_metrics.get("accuracy"))
        test_recall.append(test_metrics.get("recall"))

    return report_numbers, val_accuracy, val_recall, test_accuracy, test_recall


def main() -> int:
    report_numbers, val_accuracy, val_recall, test_accuracy, test_recall = final_metric_series()
    if not report_numbers:
        return 0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle("Training history across reports")

    axes[0].plot(report_numbers, val_accuracy, marker="o", linewidth=2, label="validation accuracy")
    axes[0].plot(report_numbers, test_accuracy, marker="o", linewidth=2, label="test accuracy")
    axes[0].set_title("Accuracy across reports")
    axes[0].set_xlabel("Report index")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(report_numbers, val_recall, marker="o", linewidth=2, label="validation recall")
    axes[1].plot(report_numbers, test_recall, marker="o", linewidth=2, label="test recall")
    axes[1].set_title("Recall across reports")
    axes[1].set_xlabel("Report index")
    axes[1].set_ylabel("Recall")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    print(f"Showing performance history from {len(report_numbers)} reports in {REPORT_DIR}...")
    plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())