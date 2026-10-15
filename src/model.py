
import torch
from torch import nn


class CandlestickCNN(nn.Module):
    def __init__(
        self,
        num_features: int = 7,
        num_classes: int = 6,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                num_features, 32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                32, 64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        # Latest and previous candle features.
        self.classifier = nn.Sequential(
            nn.Linear(64 + 2 * num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features, time)
        conv_features = self.conv(x)

        # Preserve the representation of the latest candle.
        latest_context = conv_features[:, :, -1]

        # Direct access to the last two candles.
        last_two = x[:, :, -2:].flatten(start_dim=1)

        combined = torch.cat(
            [latest_context, last_two], dim=1
        )
        return self.classifier(combined)
