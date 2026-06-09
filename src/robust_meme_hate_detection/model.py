"""CLIP dual-encoder fusion classifier for hateful-meme detection.

The architecture (report, Method / "Classifier") is an OpenCLIP ViT-B/32 image
and text encoder mapping a meme to ``v, t in R^512``, followed by a small fusion
head over the concatenation ``z = [t; v; |t - v|; t * v] in R^2048``::

    z -> LayerNorm -> Linear(2048, 512) -> GELU -> Linear(512, 1)   (one logit)

The single critical contract is that ``forward(images01, token_ids)`` takes the
image as a raw ``[0, 1]`` tensor: pixel **Normalize lives inside the model**, so
the forward pass is differentiable w.r.t. raw pixels. That is what lets white-box
attacks (:mod:`robust_meme_hate_detection.attacks`) and consistency/adversarial
training share one forward graph.

By default both encoders are frozen and only the ~1 M-parameter head is trained;
adversarial training unfreezes the vision tower (see ``configs/adversarial.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

# OpenAI/LAION CLIP normalisation statistics (ViT-B/32 laion2b checkpoint).
CLIP_MEAN: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)


class Normalize(nn.Module):
    """Differentiable per-channel normalisation of a ``(B, 3, H, W)`` tensor in ``[0, 1]``."""

    def __init__(self, mean: Sequence[float] = CLIP_MEAN, std: Sequence[float] = CLIP_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class FusionHead(nn.Module):
    """``[t; v; |t - v|; t * v] -> LayerNorm -> Linear -> GELU -> Linear`` -> 1 logit."""

    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
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
class ModelConfig:
    """Architecture + which encoders are trainable.

    ``freeze_image_encoder`` / ``freeze_text_encoder`` override the coarse
    ``freeze_encoders`` flag when set, letting adversarial training unfreeze only
    the vision tower.
    """

    arch: str = "ViT-B-32"
    pretrained: str | None = "laion2b_s34b_b79k"
    freeze_encoders: bool = True
    freeze_image_encoder: bool | None = None
    freeze_text_encoder: bool | None = None
    head_hidden: int = 512
    head_dropout: float = 0.2


class CLIPHateMemeClassifier(nn.Module):
    """OpenCLIP dual-encoder + fusion head; ``forward(images01, token_ids) -> logit``."""

    def __init__(self, config: ModelConfig | None = None, *, clip_model: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        if clip_model is None:
            import open_clip

            clip_model, _, _ = open_clip.create_model_and_transforms(
                self.config.arch, pretrained=self.config.pretrained
            )
        self.clip = clip_model
        self.embed_dim = int(getattr(self.clip.visual, "output_dim", 512))

        self.normalize = Normalize(CLIP_MEAN, CLIP_STD)
        self.head = FusionHead(self.embed_dim, hidden=self.config.head_hidden, dropout=self.config.head_dropout)

        self._apply_freezing()

    # ------------------------------------------------------------- freezing
    def _apply_freezing(self) -> None:
        c = self.config
        if c.freeze_image_encoder is None and c.freeze_text_encoder is None:
            if c.freeze_encoders:
                self.set_encoder_trainability(image_trainable=False, text_trainable=False)
        else:
            self.set_encoder_trainability(
                image_trainable=not bool(c.freeze_image_encoder) if c.freeze_image_encoder is not None else True,
                text_trainable=not bool(c.freeze_text_encoder) if c.freeze_text_encoder is not None else True,
            )

    def set_encoder_trainability(self, *, image_trainable: bool, text_trainable: bool) -> None:
        """Freeze everything, then re-enable grads for the requested tower(s)."""
        for p in self.clip.parameters():
            p.requires_grad = False
        if image_trainable:
            for p in self.clip.visual.parameters():
                p.requires_grad = True
        if text_trainable:
            for name in ("transformer", "token_embedding", "ln_final"):
                mod = getattr(self.clip, name, None)
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad = True
            for attr in ("text_projection", "positional_embedding"):
                obj = getattr(self.clip, attr, None)
                if isinstance(obj, nn.Parameter):
                    obj.requires_grad = True
                elif isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad = True

    # -------------------------------------------------------------- encoders
    def encode_image(self, images01: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(self.normalize(images01))

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(token_ids)

    # --------------------------------------------------------------- forward
    def forward(self, images01: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        v = self.encode_image(images01).float()
        t = self.encode_text(token_ids).float()
        return self.head(t, v)

    def forward_image_only(self, images01: torch.Tensor) -> torch.Tensor:
        """Image-branch-only forward (text embedding zeroed) -- used by the audit."""
        v = self.encode_image(images01).float()
        return self.head(torch.zeros_like(v), v)

    def forward_text_only(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Text-branch-only forward (image embedding zeroed) -- used by the audit."""
        t = self.encode_text(token_ids).float()
        return self.head(t, torch.zeros_like(t))

    # ---------------------------------------------------------- introspection
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- IO
# Checkpoints store the model config plus a (possibly slimmed) state dict. Frozen
# encoder weights are identical to the fresh pretrained init, so they need not be
# saved: ``save_slim_checkpoint`` keeps only the head and any *trainable* encoder,
# and ``load_classifier`` rebuilds the base model and overlays the slim weights.


def save_slim_checkpoint(model: CLIPHateMemeClassifier, path: str | Path, **extra: Any) -> None:
    """Save head + trainable-encoder weights only (frozen CLIP comes from pretrained)."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n in model.state_dict() if n.startswith("head.") or n.startswith("normalize.")}
    state = {k: v for k, v in model.state_dict().items() if k in keep}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "model_state": state, **extra}, path)


def load_classifier(path: str | Path, device: str | torch.device = "cpu") -> CLIPHateMemeClassifier:
    """Rebuild a classifier from a (full or slim) checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_cfg = ckpt.get("config", {})
    fields = ModelConfig().__dataclass_fields__
    cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in fields})
    model = CLIPHateMemeClassifier(cfg).to(device)
    # ``strict=False``: frozen encoder tensors absent from a slim dict stay at
    # their (identical) pretrained values.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model
