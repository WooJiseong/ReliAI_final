"""Custom alpha-beta-CROWN hooks for the Assignment 4 audio model."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.AudioCResNet5 import AudioCResNet5  # noqa: E402


def audio_cresnet5(num_classes=10, in_planes=8):
    """Return the PyTorch model architecture used for SpeechCommands."""
    return AudioCResNet5(num_classes=num_classes, in_planes=in_planes)


def speech_commands_fixture(spec):
    """Load a small fixed SpeechCommands feature batch for verification.

    alpha-beta-CROWN expects a callable returning:
    X, labels, data_max, data_min, eps.
    The saved fixture stores already-normalized log-mel features with shape
    (N, 1, 32, 32), so the Linf epsilon is applied directly in feature space.
    """
    fixture_path = PROJECT_ROOT / "verification" / "audio_fixture.pt"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"{fixture_path} not found. Run `python verification/create_fixture.py` first."
        )

    fixture = torch.load(fixture_path, map_location="cpu")
    X = fixture["X"].float()
    labels = fixture["labels"].long()
    eps = spec["epsilon"]
    if eps is None:
        raise ValueError("Set specification.epsilon in the YAML config.")

    # Log-mel features are normalized tensors, not bounded image pixels, so we
    # omit data_min/data_max and let alpha-beta-CROWN use X +/- epsilon.
    return {
        "X": X,
        "labels": labels,
        "eps": torch.tensor(float(eps)).reshape(1, 1, 1, 1),
        "norm": float("inf"),
        "runnerup": None,
        "target_label": None,
    }
