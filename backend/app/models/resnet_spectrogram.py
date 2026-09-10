import os
import torch
import torch.nn as nn
from typing import List


class BasicBlock2D(nn.Module):
    """
    Standard 2D Residual Block with Batch Normalization and ReLU activations
    adapted for audio spectrogram feature maps.
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet18Spectrogram(nn.Module):
    """
    ResNet-18 variant specialized for 2D Log-Mel-Spectrogram anti-spoofing feature analysis.
    Takes single-channel 2D spectrogram matrices and outputs spoof probability [0.0, 1.0].
    """

    def __init__(self, in_channels: int = 1, base_planes: int = 32) -> None:
        super().__init__()
        self.in_planes: int = base_planes

        # Spectrogram input convolution (1 channel input)
        self.conv1 = nn.Conv2d(
            in_channels,
            base_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residual stage layers
        self.layer1 = self._make_layer(base_planes, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_planes * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_planes * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_planes * 8, num_blocks=2, stride=2)

        # Global pooling and sigmoid probability head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_planes * 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock2D(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock2D.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D Log-Mel-Spectrogram.

        :param x: Tensor of shape (n_mels, time), (1, n_mels, time), or (batch, 1, n_mels, time).
        :return: Tensor of shape (batch, 1) containing spoof probability scores in [0.0, 1.0].
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_mels, time)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    @torch.no_grad()
    def predict_spoof_prob(self, spectrogram: torch.Tensor) -> float:
        """
        Inference helper returning raw scalar spoof probability for a single spectrogram.

        :param spectrogram: 2D Log-Mel-Spectrogram tensor.
        :return: Float probability score between 0.0 and 1.0.
        """
        self.eval()
        prob_tensor = self.forward(spectrogram)
        return float(prob_tensor.squeeze().item())

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_weights(self, weight_path: str) -> bool:
        """
        Load pre-trained model weights from disk onto the active device.
        Enforces eval mode following weight ingestion.

        :param weight_path: Local filesystem path to the .pth checkpoint.
        :return: True if weights were successfully loaded, False otherwise.
        """
        if not os.path.exists(weight_path):
            return False

        state_dict = torch.load(weight_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        self.eval()
        return True

