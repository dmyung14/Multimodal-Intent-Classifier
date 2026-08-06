"""
Milestone 3: MISA (Modality-Invariant and -Specific Representations) model.
=================================================================
GOAL OF THIS FILE
    Defines the MISA architecture and its four training losses, kept
    separate from the training loop (src/train_misa.py) so the model
    itself is easy to read without training-loop noise.

    MISA's core idea: instead of just concatenating text/audio/video
    features (Milestone 2's approach, which let audio/video's much larger
    raw magnitude dominate the classifier), split each modality into:
      - a SHARED ("modality-invariant") representation -- the same
        projection is applied to every modality, encouraging them to
        learn a common, comparable representation
      - a PRIVATE ("modality-specific") representation -- a separate
        projection per modality, capturing what's unique to that modality

    Three auxiliary losses shape these representations during training:
      - similarity loss: pulls the three modalities' SHARED
        representations toward each other (they should agree)
      - difference loss: pushes each modality's PRIVATE representation
        away from its own SHARED representation, and away from other
        modalities' PRIVATE representations (they should capture
        different information, not duplicate it)
      - reconstruction loss: shared+private together should be able to
        reconstruct the original encoded representation (a sanity check
        that splitting into shared/private didn't throw information away)

    This is a deliberately simplified adaptation of the original MISA
    paper (Hazarika et al. 2020) -- see
    docs/superpowers/specs/2026-08-06-milestone3-misa-fusion-design.md
    for exactly what was simplified and why (MLP encoders instead of
    BiLSTMs, since Milestone 2 already pooled each clip into one vector
    per modality; simpler similarity/difference losses than the paper's
    CMD/Frobenius-norm versions).

TERMS YOU'LL SEE
    - embedding / encoded representation : a fixed-length vector a small
      neural network produces to summarize its input
    - shared subspace / private subspace : two different "rooms" a
      modality's information gets projected into -- shared is meant to
      hold what's common across modalities, private what's unique
    - self-attention : a mechanism where each item in a set can look at
      (and be influenced by) every other item before deciding its output
    - orthogonal : at a 90-degree angle -- vectors that are orthogonal
      share no common direction, i.e. capture unrelated information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MISAModel(nn.Module):
    def __init__(self, text_dim, audio_dim=768, video_dim=512, hidden_dim=128, num_classes=20):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_encoder = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU())
        self.audio_encoder = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU())
        self.video_encoder = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.ReLU())

        # SAME projection for every modality -- this weight-sharing is what
        # makes the result "modality-invariant."
        self.shared_proj = nn.Linear(hidden_dim, hidden_dim)

        # A SEPARATE projection per modality -- this is what makes each
        # result "modality-specific."
        self.private_text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.private_audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.private_video_proj = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(2 * hidden_dim, hidden_dim)

        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=2 * hidden_dim, batch_first=True
        )
        self.fusion_layer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_feat, audio_feat, video_feat):
        """
        text_feat, audio_feat, video_feat: (batch, text_dim)/(batch, 768)/(batch, 512)
        Returns: logits (batch, num_classes), and a dict of intermediate
        representations the loss functions need.
        """
        enc_t = self.text_encoder(text_feat)
        enc_a = self.audio_encoder(audio_feat)
        enc_v = self.video_encoder(video_feat)

        shared_t = self.shared_proj(enc_t)
        shared_a = self.shared_proj(enc_a)
        shared_v = self.shared_proj(enc_v)

        private_t = self.private_text_proj(enc_t)
        private_a = self.private_audio_proj(enc_a)
        private_v = self.private_video_proj(enc_v)

        recon_t = self.decoder(torch.cat([shared_t, private_t], dim=-1))
        recon_a = self.decoder(torch.cat([shared_a, private_a], dim=-1))
        recon_v = self.decoder(torch.cat([shared_v, private_v], dim=-1))

        # Stack the 6 representations as a length-6 sequence per sample so
        # self-attention can let them influence each other, then mean-pool.
        fusion_input = torch.stack(
            [shared_t, shared_a, shared_v, private_t, private_a, private_v], dim=1
        )  # (batch, 6, hidden_dim)
        fused = self.fusion_layer(fusion_input).mean(dim=1)  # (batch, hidden_dim)

        logits = self.classifier(fused)

        reps = {
            "enc_t": enc_t, "enc_a": enc_a, "enc_v": enc_v,
            "shared_t": shared_t, "shared_a": shared_a, "shared_v": shared_v,
            "private_t": private_t, "private_a": private_a, "private_v": private_v,
            "recon_t": recon_t, "recon_a": recon_a, "recon_v": recon_v,
        }
        return logits, reps


def similarity_loss(reps):
    """Pull the three modalities' SHARED representations toward each other."""
    s_t, s_a, s_v = reps["shared_t"], reps["shared_a"], reps["shared_v"]
    return (
        F.mse_loss(s_t, s_a) + F.mse_loss(s_t, s_v) + F.mse_loss(s_a, s_v)
    ) / 3.0


def difference_loss(reps):
    """Push each modality's PRIVATE representation away from its own SHARED
    representation, and away from other modalities' PRIVATE representations.

    This is the orthogonality constraint from the glossary at the top of this
    file: squared cosine similarity is 0 exactly when two vectors sit at 90
    degrees to each other, so minimizing it pushes them toward orthogonal --
    i.e. toward carrying unrelated information."""

    def cos_sq(a, b):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return (a_norm * b_norm).sum(dim=-1).pow(2).mean()

    private_shared = (
        cos_sq(reps["private_t"], reps["shared_t"])
        + cos_sq(reps["private_a"], reps["shared_a"])
        + cos_sq(reps["private_v"], reps["shared_v"])
    ) / 3.0
    private_private = (
        cos_sq(reps["private_t"], reps["private_a"])
        + cos_sq(reps["private_t"], reps["private_v"])
        + cos_sq(reps["private_a"], reps["private_v"])
    ) / 3.0
    return private_shared + private_private


def reconstruction_loss(reps):
    """Shared+private together should reconstruct the original encoded
    representation -- a sanity check that no information was thrown away."""
    return (
        F.mse_loss(reps["recon_t"], reps["enc_t"].detach())
        + F.mse_loss(reps["recon_a"], reps["enc_a"].detach())
        + F.mse_loss(reps["recon_v"], reps["enc_v"].detach())
    ) / 3.0


def misa_loss(logits, labels, reps, sim_weight=0.5, diff_weight=0.5, recon_weight=0.5):
    """Combine task loss with the three MISA auxiliary losses."""
    task = F.cross_entropy(logits, labels)
    sim = similarity_loss(reps)
    diff = difference_loss(reps)
    recon = reconstruction_loss(reps)
    total = task + sim_weight * sim + diff_weight * diff + recon_weight * recon
    return total, {"task": task.item(), "sim": sim.item(), "diff": diff.item(), "recon": recon.item()}


if __name__ == "__main__":
    # Smoke test: random tensors of the right shapes should produce
    # correctly-shaped logits and finite losses. Run with:
    #   python src/misa_model.py
    torch.manual_seed(0)
    batch, text_dim, num_classes = 4, 1400, 20
    model = MISAModel(text_dim=text_dim, num_classes=num_classes)

    text_feat = torch.randn(batch, text_dim)
    audio_feat = torch.randn(batch, 768)
    video_feat = torch.randn(batch, 512)
    labels = torch.randint(0, num_classes, (batch,))

    logits, reps = model(text_feat, audio_feat, video_feat)
    assert logits.shape == (batch, num_classes), f"unexpected logits shape {logits.shape}"

    total_loss, parts = misa_loss(logits, labels, reps)
    assert torch.isfinite(total_loss), "loss is not finite"
    for name, value in parts.items():
        assert value == value, f"{name} loss is NaN"  # NaN != NaN

    print("Smoke test passed.")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss breakdown: {parts}, total={total_loss.item():.4f}")
