"""Create deterministic model and data artifacts for alpha-beta-CROWN."""

from pathlib import Path
import random
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocess import COMMANDS_10, SpeechCommands10, prepare_dataset  # noqa: E402
from model.AudioCResNet5 import AudioCResNet5, count_parameters  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_real_features(data_root, split, count):
    dataset = SpeechCommands10(data_root, split=split)
    xs, ys, wav_paths = [], [], []
    for index in range(min(count, len(dataset))):
        x, y = dataset[index]
        xs.append(x)
        ys.append(y)
        wav_paths.append(str(dataset.samples[index][0]))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), wav_paths


def synthetic_features(count):
    generator = torch.Generator().manual_seed(2026)
    X = torch.randn(count, 1, 32, 32, generator=generator).clamp(-2.5, 2.5)
    labels = torch.arange(count, dtype=torch.long) % 10
    return X, labels, [None] * count


def checkpoint_state_dict(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"], state
        if "model" in state:
            return state["model"], state
    return state, {}


def save_demo_checkpoint(model, checkpoint):
    with torch.no_grad():
        model.linear2.weight.zero_()
        model.linear2.bias.zero_()
        model.linear2.bias[0] = 1.0
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": (
                "yes", "no", "up", "down", "left",
                "right", "on", "off", "stop", "go",
            ),
            "input_shape": (1, 32, 32),
            "parameters": count_parameters(model),
            "source": "deterministic initialization for Assignment 4 verification demo",
        },
        checkpoint,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/SpeechCommands")
    parser.add_argument("--split", default="validation", choices=("training", "validation", "testing"))
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument("--checkpoint", default="model/AudioCResNet5.pt")
    parser.add_argument("--abcrown-checkpoint", default="model/AudioCResNet5_abcrown.pt")
    parser.add_argument("--demo-if-missing", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    model = AudioCResNet5(num_classes=10)
    source_checkpoint = (PROJECT_ROOT / args.checkpoint).resolve()
    abcrown_checkpoint = (PROJECT_ROOT / args.abcrown_checkpoint).resolve()
    abcrown_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if source_checkpoint.exists():
        state_dict, metadata = checkpoint_state_dict(source_checkpoint)
        model.load_state_dict(state_dict)
        torch.save(
            {
                **metadata,
                "state_dict": state_dict,
                "input_shape": (1, 32, 32),
                "parameters": count_parameters(model),
                "source": str(source_checkpoint),
            },
            abcrown_checkpoint,
        )
        checkpoint_source = str(source_checkpoint)
    elif abcrown_checkpoint.exists() and not args.demo_if_missing:
        state_dict, _ = checkpoint_state_dict(abcrown_checkpoint)
        model.load_state_dict(state_dict)
        checkpoint_source = str(abcrown_checkpoint)
    elif args.demo_if_missing:
        save_demo_checkpoint(model, abcrown_checkpoint)
        checkpoint_source = "deterministic demo checkpoint"
    else:
        raise FileNotFoundError(
            f"{source_checkpoint} not found. Train first or pass --demo-if-missing."
        )

    try:
        prepare_dataset(
            args.data_root,
            download=args.download,
            force_download=args.force_download,
            verify_checksum=not args.no_verify_checksum,
        )
        X, original_labels, wav_paths = load_real_features(
            args.data_root, args.split, args.count
        )
        source = f"SpeechCommands v0.02 {args.split}"
    except Exception as exc:
        reason = str(exc).strip().splitlines()[0]
        print(f"Using synthetic fixture because SpeechCommands is unavailable: {reason}")
        X, original_labels, wav_paths = synthetic_features(args.count)
        source = "synthetic log-mel-shaped fixture"

    with torch.no_grad():
        labels = model(X).argmax(dim=1)

    fixture = PROJECT_ROOT / "verification" / "audio_fixture.pt"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": X,
            "labels": labels,
            "original_labels": original_labels,
            "source": source,
            "label_semantics": "model predictions used for local stability verification",
            "class_names": COMMANDS_10,
            "wav_paths": wav_paths,
        },
        fixture,
    )
    print(f"checkpoint source: {checkpoint_source}")
    print(f"saved alpha-beta-CROWN checkpoint: {abcrown_checkpoint}")
    print(f"saved fixture: {fixture}")
    print(f"fixture source: {source}")
    print(f"fixture shape: {tuple(X.shape)} labels: {labels.tolist()}")


if __name__ == "__main__":
    main()
