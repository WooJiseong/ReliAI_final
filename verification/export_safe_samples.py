"""Export samples proven safe by alpha-beta-CROWN."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch


SAFE_PREFIXES = ("safe",)


def is_safe_status(status):
    return any(status.startswith(prefix) for prefix in SAFE_PREFIXES)


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_indices(results):
    indices = set()
    summary = results.get("summary", {})
    for status, status_indices in summary.items():
        if is_safe_status(status):
            indices.update(int(i) for i in status_indices)
    return sorted(indices)


def status_by_index(results):
    mapping = {}
    for index, item in enumerate(results.get("results", [])):
        if not item:
            continue
        mapping[index] = {
            "status": item[0],
            "time": float(item[1]) if len(item) > 1 else None,
        }
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        default="verification/audio_fixture.pt",
        help="Fixture file created by verification/create_fixture.py.",
    )
    parser.add_argument(
        "--results",
        default="verification/results/audio_cresnet5_out.pkl",
        help="alpha-beta-CROWN result pickle.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/safe_verified",
        help="Directory where safe samples and metadata will be written.",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture = torch.load(fixture_path, map_location="cpu")
    results = load_results(results_path)
    indices = safe_indices(results)
    statuses = status_by_index(results)

    X = fixture["X"]
    labels = fixture["labels"]
    original_labels = fixture.get("original_labels")

    exported = []
    for index in indices:
        sample = {
            "X": X[index].cpu(),
            "label": labels[index].cpu(),
            "original_label": (
                original_labels[index].cpu()
                if original_labels is not None
                else labels[index].cpu()
            ),
            "verification": statuses.get(index, {"status": "safe", "time": None}),
            "source": fixture.get("source", "unknown"),
            "label_semantics": fixture.get("label_semantics", "unknown"),
            "fixture_index": index,
        }
        sample_path = output_dir / f"sample_{index:04d}.pt"
        torch.save(sample, sample_path)
        exported.append(
            {
                "fixture_index": index,
                "path": str(sample_path),
                "label": int(sample["label"]),
                "original_label": int(sample["original_label"]),
                "status": sample["verification"]["status"],
                "time": sample["verification"]["time"],
            }
        )

    torch.save(
        {
            "indices": torch.tensor(indices, dtype=torch.long),
            "X": X[indices].cpu() if indices else X[:0].cpu(),
            "labels": labels[indices].cpu() if indices else labels[:0].cpu(),
            "source": fixture.get("source", "unknown"),
            "statuses": [statuses.get(index, {"status": "safe", "time": None}) for index in indices],
        },
        output_dir / "safe_samples.pt",
    )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)
        f.write("\n")

    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("fixture_index", "path", "label", "original_label", "status", "time"),
        )
        writer.writeheader()
        writer.writerows(exported)

    print(f"exported safe samples: {len(exported)}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
