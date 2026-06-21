import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.preprocess import SpeechCommands10, prepare_dataset
from model.AudioCResNet5 import AudioCResNet5, count_parameters


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset, limit):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def run_smoke_test(device):
    model = AudioCResNet5(num_classes=10).to(device)
    x = torch.randn(4, 1, 32, 32, device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)
    criterion = nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    print("smoke_test: ok")
    print(f"parameters: {count_parameters(model)}")
    print(f"input_shape: {tuple(x.shape)}")
    print(f"output_shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate AudioCResNet5 on SpeechCommands 10-command."
    )
    parser.add_argument("--data-root", default="data/SpeechCommands")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--checkpoint", default="model/AudioCResNet5.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    if args.smoke_test:
        run_smoke_test(device)
        return

    prepare_dataset(
        args.data_root,
        download=args.download,
        force_download=args.force_download,
        verify_checksum=not args.no_verify_checksum,
    )

    train_set = SpeechCommands10(args.data_root, split="training")
    val_set = SpeechCommands10(args.data_root, split="validation")
    train_set = limit_dataset(train_set, args.limit_train)
    val_set = limit_dataset(val_set, args.limit_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not args.eval_only,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AudioCResNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) else state)
        print(f"loaded checkpoint: {checkpoint}")

    print(f"device: {device}")
    print(f"parameters: {count_parameters(model)}")
    print(f"train_samples: {len(train_set)}")
    print(f"val_samples: {len(val_set)}")

    if args.eval_only:
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"eval loss={val_loss:.4f} acc={val_acc:.4f}")
        return

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "state_dict": model.state_dict(),
                    "classes": train_loader.dataset.dataset.commands
                    if isinstance(train_loader.dataset, Subset)
                    else train_loader.dataset.commands,
                    "input_shape": (1, 32, 32),
                    "parameters": count_parameters(model),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                checkpoint,
            )
            print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
