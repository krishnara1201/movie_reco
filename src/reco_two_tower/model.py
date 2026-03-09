from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, num_entities: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.user_tower = Tower(num_users, embedding_dim, hidden_dim, output_dim)
        self.item_tower = Tower(num_items, embedding_dim, hidden_dim, output_dim)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.encode_users(user_ids)
        item_vec = self.encode_items(item_ids)
        return (user_vec * item_vec).sum(dim=-1)
