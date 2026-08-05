from pathlib import Path

TFLITE_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.tflite"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.h5"
REPORT_DIR = Path(__file__).resolve().parent.parent / "model" / "report"