import json
from pathlib import Path

import numpy as np

PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"
CACHE = PROCESSED_DATA_PATH / "feature_dataset.npz"
MANIFEST = PROCESSED_DATA_PATH / "feature_dataset_manifest.json"


def main():
    print(f"Cache exists: {CACHE.exists()}")
    print(f"Manifest exists: {MANIFEST.exists()}")

    ok = True
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            print("Manifest summary:")
            print(f" - samples: {manifest.get('samples')}")
            print(f" - feature_dim: {manifest.get('feature_dim')}")
            print(f" - files entries: {len(manifest.get('files', []))}")
        except Exception as exc:
            print(f"Failed to read manifest: {exc}")
            ok = False

    if CACHE.exists():
        try:
            npz = np.load(CACHE)
            print("Cache contents:")
            print(f" - keys: {list(npz.files)}")
            if 'features' in npz:
                f = npz['features']
                print(f" - features.shape: {getattr(f, 'shape', None)}")
            if 'labels' in npz:
                l = npz['labels']
                print(f" - labels.shape: {getattr(l, 'shape', None)}, unique counts: {np.unique(l, return_counts=True)})")
        except Exception as exc:
            print(f"Failed to read cache: {exc}")
            ok = False

    # Basic consistency check to allow tasks to decide whether to re-extract
    if MANIFEST.exists() and CACHE.exists() and ok:
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            npz = np.load(CACHE)
            manifest_samples = int(manifest.get('samples', -1))
            manifest_dim = int(manifest.get('feature_dim', -1))
            features = npz['features']
            labels = npz['labels']
            if manifest_samples == int(labels.size) and manifest_dim == int(features.shape[1]):
                print("Cache and manifest validated: OK")
                return 0
            else:
                print("Cache and manifest validation failed: mismatch")
                return 1
        except Exception as exc:
            print(f"Consistency check failed: {exc}")
            return 1
    else:
        print("Cache or manifest missing or unreadable")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
