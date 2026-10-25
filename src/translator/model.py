import torch.nn as nn


class SignGRU(nn.Module):
    """(B, T, D) -> логиты (B, n_classes)."""

    def __init__(self, in_dim, n_classes, hidden=192):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        z, _ = self.gru(self.proj(x))
        return self.head(z.mean(dim=1))
