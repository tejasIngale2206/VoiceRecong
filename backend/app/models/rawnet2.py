import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class SincConv(nn.Module):
    """
    Sinc-based convolution layer directly processing raw time-domain waveforms.
    Initializes bandpass filters using the Mel scale.
    """

    @staticmethod
    def to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int = 20,
        kernel_size: int = 1024,
        in_channels: int = 1,
        sample_rate: int = 16000,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize filterbanks using Mel scale
        n_fft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(n_fft / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)

        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        window = torch.from_numpy(np.hamming(self.kernel_size)).float()
        for i in range(self.out_channels):
            fmin = filbandwidthsf[i]
            fmax = filbandwidthsf[i + 1]
            h_high = (2.0 * fmax / self.sample_rate) * np.sinc(2.0 * fmax * hsupp.numpy() / self.sample_rate)
            h_low = (2.0 * fmin / self.sample_rate) * np.sinc(2.0 * fmin * hsupp.numpy() / self.sample_rate)
            hideal = torch.from_numpy(h_high - h_low).float()
            band_pass[i, :] = window * hideal

        # Non-persistent buffer: automatically transfers to GPU/CPU with .to()
        # while keeping state_dict identical to official ASVspoof checkpoints
        self.register_buffer(
            "filters",
            band_pass.view(self.out_channels, 1, self.kernel_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
        )


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization, LeakyReLU, and MaxPool
    matching the official ASVspoof RawNet2 architecture.
    """

    def __init__(self, in_channels: int, out_channels: int, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(in_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.downsample = in_channels != out_channels
        if self.downsample:
            self.conv_downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x if self.first else self.lrelu(self.bn1(x))
        out = self.conv2(self.lrelu(self.bn2(self.conv1(out))))
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return self.mp(out)


class RawNet2(nn.Module):
    """
    Official RawNet2 anti-spoofing architecture (ASVspoof 2019/2021 DF Baseline).
    Processes 1D raw 16kHz audio waveforms through SincNet, residual blocks with
    Feature Map Scaling (FMS), and a recurrent GRU classifier head.
    Outputs 2-class logits: [logit_bonafide, logit_spoof].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 2,
    ) -> None:
        super().__init__()
        self.Sinc_conv = SincConv(out_channels=20, kernel_size=1024, in_channels=in_channels)
        self.first_bn = nn.BatchNorm1d(20)
        self.selu = nn.SELU(inplace=True)

        self.block0 = nn.Sequential(ResidualBlock1D(20, 20, first=True))
        self.block1 = nn.Sequential(ResidualBlock1D(20, 20))
        self.block2 = nn.Sequential(ResidualBlock1D(20, 128))
        self.block3 = nn.Sequential(ResidualBlock1D(128, 128))
        self.block4 = nn.Sequential(ResidualBlock1D(128, 128))
        self.block5 = nn.Sequential(ResidualBlock1D(128, 128))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = nn.Sequential(nn.Linear(20, 20))
        self.fc_attention1 = nn.Sequential(nn.Linear(20, 20))
        self.fc_attention2 = nn.Sequential(nn.Linear(128, 128))
        self.fc_attention3 = nn.Sequential(nn.Linear(128, 128))
        self.fc_attention4 = nn.Sequential(nn.Linear(128, 128))
        self.fc_attention5 = nn.Sequential(nn.Linear(128, 128))

        self.bn_before_gru = nn.BatchNorm1d(128)
        self.gru = nn.GRU(input_size=128, hidden_size=1024, num_layers=3, batch_first=True)
        self.fc1_gru = nn.Linear(1024, 1024)
        self.fc2_gru = nn.Linear(1024, out_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time-domain waveform tensor.

        :param x: Tensor of shape (samples,), (batch, samples), or (batch, 1, samples).
        :return: Logits tensor of shape (batch, 2) where Index 0=Bona Fide, Index 1=Spoof.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        blocks = [
            (self.block0, self.fc_attention0),
            (self.block1, self.fc_attention1),
            (self.block2, self.fc_attention2),
            (self.block3, self.fc_attention3),
            (self.block4, self.fc_attention4),
            (self.block5, self.fc_attention5),
        ]

        for block, att in blocks:
            x_b = block(x)
            y = self.avgpool(x_b).squeeze(-1)
            y = self.sig(att(y)).unsqueeze(-1)
            x = x_b * y + y

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        return self.fc2_gru(x)

    @torch.no_grad()
    def predict_spoof_prob(self, waveform: torch.Tensor) -> float:
        """
        Inference helper returning raw scalar spoof probability for a single chunk.
        Official ASVspoof mapping: Index 0 = Bona Fide, Index 1 = Spoof.
        """
        self.eval()
        wave_dev = waveform.to(self.device)
        logits = self.forward(wave_dev)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0][1].item())

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

        try:
            state_dict = torch.load(weight_path, map_location=self.device)
            state_dict = (
                state_dict.get("model_state_dict", state_dict)
                if isinstance(state_dict, dict)
                else state_dict
            )
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(cleaned, strict=False)
            self.eval()
            return True
        except Exception:
            return False
