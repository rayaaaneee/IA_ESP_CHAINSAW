from .globals import REPORT_DIR, discover_reports
from .plot_history import main as plot_history
from .plot_latest_report import main as plot_latest_report

__all__ = ["REPORT_DIR", "discover_reports", "plot_history", "plot_latest_report"]
