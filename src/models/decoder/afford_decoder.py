from typing import Tuple

import torch
from torch import nn


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        downsample_rate: int = 8,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        hidden_dim = transformer_dim // downsample_rate
        self.out_head = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            SwapAexes(),
            nn.GroupNorm(4, hidden_dim),
            SwapAexes(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        point_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        point_pes: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = point_embeddings
        tokens = prompt_embeddings
        hs, src = self.transformer(src, tokens, point_pes)
        affordance = self.out_head(src)
        masks = self.sigmoid(affordance)
        
        return masks


class SwapAexes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)
    
