import json
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "data" / "processed" / "feature_dataset_manifest.json"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files", [])

    counts_by_folder = defaultdict(Counter)
    unexpected = []

    for entry in files:
        f = entry.get("file", "")
        label = entry.get("label")
        top = Path(f).parts[0] if f else ""
        counts_by_folder[top][label] += entry.get("windows", 1)
        # simple heuristic: if top folder name suggests environment but label==1, flag it
        if top and top.lower() != "chainsaw" and label == 1:
            unexpected.append((f, label))

    print("Label counts by top folder (sum of windows):")
    for folder, ctr in counts_by_folder.items():
        print(f" - {folder}: {{}}".format(dict(ctr)))

    if unexpected:
        print("\nPotentially mislabelled files (top-folder != 'chainsaw' but label==1):")
        for f, label in unexpected[:50]:
            print(f" - {f}: {label}")
    else:
        print("\nNo obvious mislabelling detected by heuristic.")


if __name__ == "__main__":
    main()
