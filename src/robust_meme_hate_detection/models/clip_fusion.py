"""CLIP-based dual-encoder fusion classifier for Hateful Memes.

See ``project_planning/model_architecture.md`` §3 for the full spec. The
critical contract is that ``forward(images01, token_ids)`` accepts an image
tensor in ``[0, 1]`` so that white-box attacks (FGSM/PGD) can backprop to
pixel space; Normalize lives inside the model.

We use OpenCLIP's high-level ``encode_text`` / ``encode_image`` so we are not
coupled to specific internal attribute names that vary across versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

# OpenAI CLIP normalisation statistics (laion2b/openai checkpoints).
CLIP_MEAN: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)


class Normalize(nn.Module):
    """Differentiable per-channel Normalize that operates on (B, 3, H, W) in [0, 1]."""

    def __init__(self, mean: Sequence[float] = CLIP_MEAN, std: Sequence[float] = CLIP_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class FusionHead(nn.Module):
    """concat(t, v, |t - v|, t * v) -> LayerNorm -> MLP -> 1 logit."""

    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        f = torch.cat([t, v, (t - v).abs(), t * v], dim=-1)
        return self.mlp(self.norm(f)).squeeze(-1)  # (B,) raw logit


@dataclass
class CLIPFusionConfig:
    arch: str = "ViT-B-32"
    pretrained: str | None = "laion2b_s34b_b79k"
    freeze_encoders: bool = True
    head_hidden: int = 512
    head_dropout: float = 0.2


class CLIPHateMemeClassifier(nn.Module):
    """Dual-encoder CLIP classifier with a small fusion head.

    The forward pass accepts ``images01`` (raw [0, 1] tensor of shape
    ``(B, 3, 224, 224)``) and ``token_ids`` (long tensor of shape ``(B, 77)``)
    and returns a single raw logit per example.
    """

    def __init__(
        self,
        arch: str = "ViT-B-32",
        pretrained: str | None = "laion2b_s34b_b79k",
        freeze_encoders: bool = True,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
        *,
        clip_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = CLIPFusionConfig(
            arch=arch,
            pretrained=pretrained,
            freeze_encoders=freeze_encoders,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )

        if clip_model is None:
            import open_clip

            clip_model, _, _ = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
        self.clip = clip_model

        # Determine the embedding dimension. OpenCLIP exposes both visual.output_dim
        # and text-encoder dim through the projection. We trust visual.output_dim.
        embed_dim = getattr(self.clip.visual, "output_dim", None)
        if embed_dim is None:
            # Fallback: probe with a dummy forward.
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat = self.clip.encode_image(self.normalize_dummy(dummy))
                embed_dim = int(feat.shape[-1])
        self.embed_dim = int(embed_dim)

        self.normalize = Normalize(CLIP_MEAN, CLIP_STD)
        self.head = FusionHead(dim=self.embed_dim, hidden=head_hidden, dropout=head_dropout)

        if freeze_encoders:
            self.freeze_encoders()

    # ------------------------------------------------------------------ helpers
    def normalize_dummy(self, x: torch.Tensor) -> torch.Tensor:
        """Used only by the embed_dim fallback above."""
        return Normalize(CLIP_MEAN, CLIP_STD)(x)

    def freeze_encoders(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 2) -> None:
        """Stage-2 default: unfreeze the last ``n_blocks`` of each encoder.

        Best-effort: walks the visual transformer's residual blocks and the
        text transformer's residual blocks. Also unfreezes ``ln_final`` and
        ``text_projection`` if present.
        """
        # Visual side
        v_blocks = self._visual_blocks()
        for blk in v_blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        # Text side
        t_blocks = self._text_blocks()
        for blk in t_blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        # Final norms / projections
        for name in ("ln_final",):
            mod = getattr(self.clip, name, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True
        tp = getattr(self.clip, "text_projection", None)
        if isinstance(tp, nn.Parameter):
            tp.requires_grad = True
        elif isinstance(tp, nn.Module):
            for p in tp.parameters():
                p.requires_grad = True

    def _visual_blocks(self) -> list[nn.Module]:
        v = self.clip.visual
        for attr in ("transformer", "trunk"):
            t = getattr(v, attr, None)
            if t is not None and hasattr(t, "resblocks"):
                return list(t.resblocks)
        # Some OpenCLIP variants expose `blocks` directly.
        if hasattr(v, "blocks"):
            return list(v.blocks)
        raise RuntimeError("Could not locate visual transformer blocks for unfreezing")

    def _text_blocks(self) -> list[nn.Module]:
        t = getattr(self.clip, "transformer", None)
        if t is None:
            raise RuntimeError("Could not locate text transformer for unfreezing")
        if hasattr(t, "resblocks"):
            return list(t.resblocks)
        if hasattr(t, "blocks"):
            return list(t.blocks)
        raise RuntimeError("Could not locate text transformer blocks for unfreezing")

    # ----------------------------------------------------------------- encoders
    def encode_image(self, images01: torch.Tensor) -> torch.Tensor:
        x = self.normalize(images01)
        return self.clip.encode_image(x)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(token_ids)

    # ----------------------------------------------------------------- forward
    def forward(self, images01: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        v = self.encode_image(images01)
        t = self.encode_text(token_ids)
        # OpenCLIP 2.x returns embeddings in mixed precision sometimes; cast for the head.
        return self.head(t.float(), v.float())

    def forward_text_only(self, token_ids: torch.Tensor) -> torch.Tensor:
        t = self.encode_text(token_ids).float()
        v = torch.zeros_like(t)
        return self.head(t, v)

    def forward_image_only(self, images01: torch.Tensor) -> torch.Tensor:
        v = self.encode_image(images01).float()
        t = torch.zeros_like(v)
        return self.head(t, v)

    # ----------------------------------------------------------- introspection
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_random_clip_for_testing(arch: str = "ViT-B-32", embed_dim: int = 512) -> nn.Module:
    """Return a tiny randomly-initialised OpenCLIP model for synthetic tests.

    Avoids the network download in unit tests. Falls back to a hand-rolled stub
    if OpenCLIP refuses to construct without pretrained weights.
    """
    import open_clip

    try:
        model = open_clip.create_model(arch, pretrained=None)
        return model
    except Exception:  # pragma: no cover
        # If OpenCLIP can't be created without weights, build a minimal stub.
        return _MinimalClipStub(embed_dim=embed_dim)


class _MinimalClipStub(nn.Module):
    """Minimal stand-in that exposes encode_image / encode_text and visual.output_dim."""

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, embed_dim),
        )
        # Mark the output_dim attribute the wrapper inspects.
        self.visual.output_dim = embed_dim
        self.token_embedding = nn.Embedding(49408, 64)
        self.text_proj = nn.Linear(64, embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.token_embedding(ids).mean(dim=1)
        return self.text_proj(emb)
