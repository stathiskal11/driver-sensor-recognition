from __future__ import annotations

"""Baseline joint-embedding model used for the paper reproduction."""

import torch
from torch import nn


class VisualJointEmbedding3DCNN(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        # This branch compresses the scene/gaze clip into one visual embedding.
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, scene_gaze: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(scene_gaze))


class SignalEmbeddingMLP(nn.Module):
    def __init__(self, timesteps: int = 30, signal_dim: int = 6) -> None:
        super().__init__()
        # The signal stream is flattened and modeled as dense tabular input.
        self.network = nn.Sequential(
            nn.Linear(timesteps * signal_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        flattened = signals.flatten(start_dim=1)
        return self.network(flattened)


class PaperTakeoverBaselineModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        timesteps: int = 30,
        signal_dim: int = 6,
        hmi_dim: int = 9,
    ) -> None:
        super().__init__()
        self.visual_branch = VisualJointEmbedding3DCNN(input_channels=input_channels)
        self.signal_branch = SignalEmbeddingMLP(
            timesteps=timesteps, signal_dim=signal_dim
        )
        fused_dim = 256 + 32 + hmi_dim
        # Final prediction uses visual, signal, and HMI context together.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        scene_gaze: torch.Tensor,
        signals: torch.Tensor,
        hmi: torch.Tensor,
    ) -> torch.Tensor:
        visual_embedding = self.visual_branch(scene_gaze)
        signal_embedding = self.signal_branch(signals)
        fused = torch.cat([visual_embedding, signal_embedding, hmi], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        scene_gaze: torch.Tensor,
        signals: torch.Tensor,
        hmi: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self(scene_gaze=scene_gaze, signals=signals, hmi=hmi))
