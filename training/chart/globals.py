import json
import re
from pathlib import Path
from typing import Any

REPORT_DIR = Path(__file__).resolve().parent.parent / "model" / "report"
REPORT_RE = re.compile(r"training_report_(\d+)\.json$")

def report_index(path: Path) -> int:
    match = REPORT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def discover_reports() -> list[Path]:
    return sorted(REPORT_DIR.glob("training_report_*.json"), key=report_index)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def history_series(report: dict[str, Any], key: str) -> list[float]:
    history = report.get("history", {})
    return [float(value) for value in history.get(key, [])]


