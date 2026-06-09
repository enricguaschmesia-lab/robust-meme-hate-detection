"""Model wiring tests -- CPU only, no network (a tiny stub stands in for CLIP)."""

from __future__ import annotations

import torch
import torch.nn as nn

from robust_meme_hate_detection.model import (
    CLIPHateMemeClassifier,
    FusionHead,
    ModelConfig,
    Normalize,
    save_slim_checkpoint,
)


class StubClip(nn.Module):
    """Minimal CLIP stand-in exposing the attributes the wrapper relies on."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.visual = nn.Linear(3 * 224 * 224, dim)
        self.visual.output_dim = dim
        self.text = nn.Embedding(100, dim)

    def encode_image(self, x):
        return self.visual(x.flatten(1))

    def encode_text(self, ids):
        return self.text(ids).mean(dim=1)


def _model() -> CLIPHateMemeClassifier:
    return CLIPHateMemeClassifier(ModelConfig(freeze_encoders=True), clip_model=StubClip())


def test_fusion_head_logit_shape():
    head = FusionHead(dim=8, hidden=16, dropout=0.0)
    assert head(torch.randn(3, 8), torch.randn(3, 8)).shape == (3,)


def test_normalize_preserves_shape():
    x = torch.rand(2, 3, 4, 4)
    assert Normalize()(x).shape == x.shape


def test_forward_modes_shapes():
    model = _model()
    images01 = torch.rand(4, 3, 224, 224)
    tokens = torch.randint(0, 100, (4, 7))
    assert model(images01, tokens).shape == (4,)
    assert model.forward_image_only(images01).shape == (4,)
    assert model.forward_text_only(tokens).shape == (4,)


def test_frozen_encoder_means_head_only_trainable():
    model = _model()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head.") for n in trainable)


def test_slim_checkpoint_roundtrip(tmp_path):
    model = _model()
    path = tmp_path / "slim.pt"
    save_slim_checkpoint(model, path)
    state = torch.load(path, weights_only=False)["model_state"]
    # Frozen-encoder checkpoint stores only head / normalize tensors.
    assert state and all(k.startswith(("head.", "normalize.")) for k in state)
    # Loading it back into a fresh model restores the head exactly.
    other = CLIPHateMemeClassifier(ModelConfig(freeze_encoders=True), clip_model=StubClip())
    other.load_state_dict(state, strict=False)
    for (n, a), (_, b) in zip(model.head.named_parameters(), other.head.named_parameters()):
        assert torch.equal(a, b), n
