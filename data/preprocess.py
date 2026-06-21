import argparse
import hashlib
import tarfile
import urllib.request
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import torchaudio
except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover.
    torchaudio = None
    TORCHAUDIO_IMPORT_ERROR = exc
else:
    TORCHAUDIO_IMPORT_ERROR = None


SPEECH_COMMANDS_URL = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)
ARCHIVE_SHA256 = (
    "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
)
COMMANDS_10 = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


def download_speech_commands(root, force=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "speech_commands_v0.02.tar.gz"

    if archive.exists() and not force:
        return archive

    print(f"Downloading SpeechCommands v0.02 to {archive}")
    urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive)
    return archive


def sha256sum(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_speech_commands(root, verify_checksum=True):
    root = Path(root)
    archive = root / "speech_commands_v0.02.tar.gz"
    marker = root / ".speech_commands_v0.02_extracted"

    if marker.exists():
        return
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} not found. Run with --download or place the archive there."
        )
    if verify_checksum:
        actual = sha256sum(archive)
        if actual != ARCHIVE_SHA256:
            raise RuntimeError(
                f"Unexpected archive checksum: {actual}. "
                "Use --no-verify-checksum if you intentionally use another archive."
            )

    print(f"Extracting {archive} to {root}")
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, root)
    marker.write_text("ok\n")


def safe_extract(tar, path):
    target_root = Path(path).resolve()
    for member in tar.getmembers():
        member_path = (target_root / member.name).resolve()
        if target_root != member_path and target_root not in member_path.parents:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    tar.extractall(target_root)


def _read_split_file(root, name):
    split_path = Path(root) / name
    if not split_path.exists():
        return set()
    return {line.strip() for line in split_path.read_text().splitlines() if line.strip()}


def _relative_wavs(root):
    root = Path(root)
    wavs = []
    for path in root.glob("*/*.wav"):
        label = path.parent.name
        if label == "_background_noise_":
            continue
        wavs.append(path.relative_to(root).as_posix())
    return sorted(wavs)


def list_speech_commands(root, split, commands=COMMANDS_10):
    root = Path(root)
    validation = _read_split_file(root, "validation_list.txt")
    testing = _read_split_file(root, "testing_list.txt")
    commands = set(commands)
    selected = []

    for rel_path in _relative_wavs(root):
        label = rel_path.split("/", 1)[0]
        if label not in commands:
            continue
        if split == "training" and rel_path in validation.union(testing):
            continue
        if split == "validation" and rel_path not in validation:
            continue
        if split == "testing" and rel_path not in testing:
            continue
        selected.append((root / rel_path, label))

    if not selected:
        raise RuntimeError(
            f"No SpeechCommands samples found for split={split!r} in {root}."
        )
    return selected


def require_torchaudio():
    if torchaudio is None:
        raise ImportError(
            "torchaudio is required. Install dependencies with "
            "`pip install -r requirements.txt`. If torchcodec reports missing "
            "libav* libraries, install FFmpeg runtime libraries with your system "
            "package manager."
        )


def load_wav(path):
    require_torchaudio()
    try:
        return torchaudio.load(str(path))
    except (ImportError, OSError, RuntimeError):
        return load_pcm_wav(path)


def load_pcm_wav(path):
    """Load PCM WAV without torchcodec/FFmpeg.

    SpeechCommands v0.02 files are 16-bit PCM WAV, so this fallback keeps the
    dataset usable even when torchaudio's torchcodec backend cannot load FFmpeg.
    """
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        dtype = torch.int16
        scale = 32768.0
    elif sample_width == 1:
        dtype = torch.uint8
        scale = 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

    waveform = torch.frombuffer(bytearray(frames), dtype=dtype).float()
    if sample_width == 1:
        waveform = waveform - 128.0
    waveform = waveform / scale
    waveform = waveform.view(-1, channels).t().contiguous()
    return waveform, sample_rate


class LogMelTransform:
    def __init__(
        self,
        sample_rate=16000,
        clip_seconds=1.0,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=32,
        feature_size=32,
    ):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * clip_seconds)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.feature_size = feature_size
        require_torchaudio()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=20.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )

    def _fix_length(self, waveform):
        if waveform.numel() > self.num_samples:
            return waveform[: self.num_samples]
        if waveform.numel() < self.num_samples:
            return F.pad(waveform, (0, self.num_samples - waveform.numel()))
        return waveform

    def __call__(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        waveform = self._fix_length(waveform.flatten())
        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-6)
        log_mel = F.interpolate(
            log_mel.unsqueeze(0).unsqueeze(0),
            size=(self.n_mels, self.feature_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mean = log_mel.mean()
        std = log_mel.std().clamp_min(1e-5)
        return (log_mel - mean) / std


class SpeechCommands10(Dataset):
    def __init__(self, root, split="training", transform=None, commands=COMMANDS_10):
        if split not in {"training", "validation", "testing"}:
            raise ValueError("split must be one of: training, validation, testing")
        self.root = Path(root)
        self.split = split
        self.commands = tuple(commands)
        self.class_to_idx = {label: i for i, label in enumerate(self.commands)}
        self.samples = list_speech_commands(self.root, split, self.commands)
        self.transform = transform if transform is not None else LogMelTransform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        waveform, sample_rate = load_wav(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        features = self.transform(waveform, sample_rate)
        target = self.class_to_idx[label]
        return features, target


def prepare_dataset(root, download=False, force_download=False, verify_checksum=True):
    if download:
        download_speech_commands(root, force=force_download)
    extract_speech_commands(root, verify_checksum=verify_checksum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/SpeechCommands")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument(
        "--split", default="training", choices=("training", "validation", "testing")
    )
    args = parser.parse_args()

    prepare_dataset(
        args.root,
        download=args.download,
        force_download=args.force_download,
        verify_checksum=not args.no_verify_checksum,
    )
    dataset = SpeechCommands10(args.root, split=args.split)
    x, y = dataset[0]
    print(f"split: {args.split}")
    print(f"samples: {len(dataset)}")
    print(f"classes: {dataset.commands}")
    print(f"sample feature shape: {tuple(x.shape)}")
    print(f"sample target: {y}")


if __name__ == "__main__":
    main()
