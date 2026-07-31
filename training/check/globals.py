from pathlib import Path

PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"
CACHE = PROCESSED_DATA_PATH / "feature_dataset.npz"
MANIFEST = PROCESSED_DATA_PATH / "feature_dataset_manifest.json"