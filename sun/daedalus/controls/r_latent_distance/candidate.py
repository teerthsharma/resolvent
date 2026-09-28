"""Honest ranker control: D-JEPA's native latent-distance planning, score = -|zhat_k - g| (g = 0). No parameters."""
import torch.nn as nn


class Dist(nn.Module):
    def forward(self, z):
        return -z.norm(dim=-1)


def build(spec):
    return Dist()
