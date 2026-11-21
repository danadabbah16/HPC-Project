import torch
from torch import nn


class SleepCNN(nn.Module):
    """1D CNN: one 30-s EEG epoch (n_channels, 3000) -> 5 sleep-stage scores."""

    def __init__(self, n_classes=5, n_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            # Wide first filter (0.5 s) to catch whole waves
            nn.Conv1d(n_channels, 32, kernel_size=50, stride=6),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Dropout(0.5),
            # Three smaller layers that combine patterns
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),   # average over time -> one value per filter
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    