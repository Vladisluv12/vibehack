"""Temporal classifiers with a shared (batch, time, features) interface."""
import math

import torch
import torch.nn as nn


class SignGRU(nn.Module):
    """Defaults preserve the original model and checkpoint keys."""

    def __init__(self, in_dim, n_classes, hidden=192, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        z, _ = self.gru(self.proj(x))
        return self.head(z.mean(dim=1))


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        return (x * self.score(x).softmax(dim=1)).sum(dim=1)


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dropout, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2))


class SignAttentionGRU(nn.Module):
    """Residual temporal convolutions -> BiGRU -> attention + mean + max pooling."""

    def __init__(self, in_dim, n_classes, hidden=192, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.Sequential(TemporalBlock(hidden, dropout, 1), TemporalBlock(hidden, dropout, 2))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.norm = nn.LayerNorm(2 * hidden)
        self.pool = AttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.LayerNorm(6 * hidden), nn.Dropout(dropout),
                                  nn.Linear(6 * hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        z, _ = self.gru(self.temporal(self.proj(x)))
        z = self.norm(z)
        return self.head(torch.cat([self.pool(z), z.mean(dim=1), z.amax(dim=1)], dim=-1))


class SignTransformer(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=192, layers=2, dropout=0.3, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads for Transformer")
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout,
                                       activation="gelu", batch_first=True, norm_first=True),
            num_layers=layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        # Encoder clones layers; initialize each layer independently.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        z = self.proj(x)
        position = torch.arange(z.size(1), device=z.device, dtype=z.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, z.size(2), 2, device=z.device, dtype=z.dtype)
                              * (-math.log(10000.0) / z.size(2)))
        encoding = torch.zeros_like(z[0])
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:encoding[:, 1::2].size(1)])
        z = self.encoder(z + encoding)
        return self.head(torch.cat([self.pool(z), z.mean(dim=1)], dim=-1))


ARCHITECTURES = ("gru", "attn_gru", "transformer")


def build_model(config, in_dim, n_classes):
    options = {k: config.get(k, default) for k, default in
               (("hidden", 192), ("layers", 2), ("dropout", 0.3))}
    arch = config.get("arch", "gru")
    if arch == "gru":
        return SignGRU(in_dim, n_classes, **options)
    if arch == "attn_gru":
        return SignAttentionGRU(in_dim, n_classes, **options)
    if arch == "transformer":
        return SignTransformer(in_dim, n_classes, heads=config.get("heads", 4), **options)
    raise ValueError(f"Unknown architecture: {arch}")


def model_from_checkpoint(checkpoint):
    config = checkpoint.get("model_config")
    if config is None:
        config = {"arch": "gru", "hidden": checkpoint["hidden"]}
    model = build_model(config, checkpoint["in_dim"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state"])
    return model
