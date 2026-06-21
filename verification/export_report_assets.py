"""Export WAV and PNG artifacts for reporting verifier outcomes."""

import argparse
import csv
import json
import pickle
import re
import shutil
from pathlib import Path

import torch


def status_category(status):
    status = str(status).lower()
    if status.startswith("safe"):
        return "safe"
    if "unsafe" in status or status in {"sat", "falsified"}:
        return "unsafe"
    if "unknown" in status or "timeout" in status or "timed out" in status:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "-", status).strip("-") or "unknown"


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_epsilon(config_path):
    text = Path(config_path).read_text(encoding="utf-8")
    match = re.search(r"^\s*epsilon\s*:\s*([^\s#]+)", text, flags=re.MULTILINE)
    return match.group(1) if match else "eps"


def status_by_index(results):
    mapping = {}
    for status, indices in results.get("summary", {}).items():
        for index in indices:
            mapping[int(index)] = {"status": status, "category": status_category(status)}

    for index, item in enumerate(results.get("results", [])):
        if not item:
            continue
        entry = mapping.setdefault(
            index, {"status": item[0], "category": status_category(item[0])}
        )
        entry["time"] = float(item[1]) if len(item) > 1 else None
    return mapping


def save_feature_png(feature, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = feature.squeeze(0).detach().cpu()
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=160)
    im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time bin")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="verification/audio_fixture.pt")
    parser.add_argument("--results", default="verification/results/audio_cresnet5_out.pkl")
    parser.add_argument("--config", default="verification/audio_cresnet5.yaml")
    parser.add_argument("--output-dir", default="output/report_assets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.wav", "*.png", "manifest.json", "manifest.csv"):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()

    fixture = torch.load(args.fixture, map_location="cpu")
    results = load_results(args.results)
    statuses = status_by_index(results)
    epsilon = sanitize(read_epsilon(args.config))

    X = fixture["X"]
    labels = fixture["labels"]
    original_labels = fixture.get("original_labels", labels)
    class_names = list(fixture.get("class_names", []))
    wav_paths = fixture.get("wav_paths", [None] * len(labels))

    exported = []
    for index in sorted(statuses):
        if index >= len(labels):
            continue
        label = int(labels[index])
        original_label = int(original_labels[index])
        class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        category = statuses[index]["category"]
        stem = f"{sanitize(class_name)}_{epsilon}_{category}_{index:04d}"

        png_path = output_dir / f"{stem}.png"
        save_feature_png(
            X[index],
            png_path,
            f"{class_name} eps={epsilon} {category}",
        )

        wav_path = None
        source_wav = wav_paths[index] if index < len(wav_paths) else None
        if source_wav:
            source_wav = Path(source_wav)
            if source_wav.exists():
                wav_path = output_dir / f"{stem}.wav"
                shutil.copy2(source_wav, wav_path)

        exported.append(
            {
                "fixture_index": index,
                "class": class_name,
                "label": label,
                "original_label": original_label,
                "epsilon": epsilon,
                "status": statuses[index]["status"],
                "category": category,
                "time": statuses[index].get("time"),
                "wav": str(wav_path) if wav_path is not None else "",
                "png": str(png_path),
            }
        )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)
        f.write("\n")

    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = (
            "fixture_index", "class", "label", "original_label", "epsilon",
            "status", "category", "time", "wav", "png",
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported)

    print(f"exported report assets: {len(exported)}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
