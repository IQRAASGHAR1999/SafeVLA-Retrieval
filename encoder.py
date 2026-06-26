"""
encoder.py — Lightweight multimodal scene encoder for driving scenes.

Produces a joint embedding of (front-camera image, optional text command)
that is used both for retrieval against a memory bank and for OOD scoring.

Design constraints:
- The vision backbone is FROZEN (no gradients) so the whole thing fits in
  4 GB VRAM and trains in minutes. Only a small projection head is trained.
- Works with a small CNN backbone by default (resnet18) so it runs even
  without downloading large VLM weights. Optionally swappable for a CLIP
  image encoder when more VRAM is available.

The embedding is L2-normalised so cosine similarity == dot product, which
makes nearest-neighbour retrieval a single matrix multiply.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneEncoder(nn.Module):
    """Frozen vision backbone + trainable projection head.

    Input:  images [B, 3, H, W]  (and optionally text embeddings [B, D_text])
    Output: L2-normalised joint embedding [B, embed_dim]
    """

    def __init__(self, embed_dim: int = 128, text_dim: int = 0,
                 backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_dim = text_dim

        # Frozen vision backbone
        if backbone == "resnet18":
            try:
                from torchvision.models import resnet18, ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                net = resnet18(weights=weights)
            except Exception:
                from torchvision.models import resnet18
                net = resnet18(weights=None)
            feat_dim = net.fc.in_features        # 512
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable projection head (this is the ONLY trained part)
        in_dim = feat_dim + text_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    @torch.no_grad()
    def _backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        return self.backbone(images)

    def forward(self, images: torch.Tensor,
                text_emb: torch.Tensor | None = None) -> torch.Tensor:
        feats = self._backbone_features(images)         # [B, 512]
        if self.text_dim > 0:
            if text_emb is None:
                text_emb = torch.zeros(images.shape[0], self.text_dim,
                                       device=images.device)
            feats = torch.cat([feats, text_emb], dim=-1)
        emb = self.proj(feats)                          # [B, embed_dim]
        return F.normalize(emb, dim=-1)


def _smoke_test() -> None:
    enc = SceneEncoder(embed_dim=128, text_dim=0, pretrained=False)
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable/1e3:.1f}k")
    print(f"Frozen params:    {n_frozen/1e6:.2f}M")
    x = torch.randn(4, 3, 224, 224)
    emb = enc(x)
    print(f"Embedding shape:  {tuple(emb.shape)}")
    print(f"L2 norms (should be ~1.0): {emb.norm(dim=-1).tolist()}")


if __name__ == "__main__":
    _smoke_test()
