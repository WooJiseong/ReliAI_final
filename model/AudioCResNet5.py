import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class AudioCResNet5(nn.Module):
    """Small verifier-friendly ResNet for SpeechCommands log-mel features.

    The model mirrors the CIFAR-10 resnet2b scale: 5 convolutional layers,
    2 linear layers, ReLU activations, and no batch normalization. The expected
    input shape is (N, 1, 32, 32).
    """

    def __init__(self, num_classes=10, in_planes=8):
        super().__init__()
        self.num_classes = num_classes
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_planes, in_planes * 2, stride=2),
            BasicBlock(in_planes * 2, in_planes * 2, stride=1),
        )
        self.linear1 = nn.Linear(in_planes * 2 * 8 * 8, 100)
        self.linear2 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        return self.linear2(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = AudioCResNet5(num_classes=10)
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(model)
    print("parameters:", count_parameters(model))
    print("output shape:", tuple(y.shape))
