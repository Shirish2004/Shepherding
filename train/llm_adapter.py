"""Adapter training between fixed LLM intent one-hot and dense embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class IntentAdapter(nn.Module):
    """Compact adapter for intent embedding fine-tuning."""

    def __init__(self, in_dim: int, out_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map one-hot intent to latent embedding."""

        return self.net(x)


def main() -> None:
    """Train adapter from synthetic targets for scaffold reproducibility."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("intent_adapter.pt"))
    args = parser.parse_args()
    x = torch.eye(6)
    y = torch.randn(6, 16)
    model = IntentAdapter(6)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for _ in range(200):
        pred = model(x)
        loss = mse(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
