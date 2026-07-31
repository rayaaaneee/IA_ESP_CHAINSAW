from __future__ import annotations

import argparse

from chart import (REPORT_DIR, discover_reports, plot_history,
                   plot_latest_report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training report metrics.")
    parser.add_argument("--latest", action="store_true", help="Plot the latest report curves")
    parser.add_argument("--history", action="store_true", help="Plot metrics across all reports")
    args = parser.parse_args()

    if args.latest:
        report_path = plot_latest_report()
        if report_path is None:
            print(f"No training reports found in {REPORT_DIR}.")
            return 1
        return 0

    if args.history:
        if not discover_reports():
            print(f"No training reports found in {REPORT_DIR}.")
            return 1
        return plot_history()

    parser.error("choose either --latest or --history")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
