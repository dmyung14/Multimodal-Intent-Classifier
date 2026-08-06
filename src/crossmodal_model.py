"""
Milestone 4: cross-modal transformer model for dialogue-aware intent
classification.
=================================================================
GOAL OF THIS FILE
    Milestone 3's MISA used self-attention over 6 FIXED representations
    (3 shared + 3 private) of a single utterance. This file implements
    something different: CROSS-modal attention, where one modality's
    sequence directly attends over another modality's sequence, plus
    DIALOGUE context -- the model sees not just the current utterance but
    up to 4 utterances before it in the same conversation.

    Simplified from the original "Multimodal Transformer" (MulT, Tsai et
    al. 2019): only audio->text and video->text cross-attention are
    implemented (not all 6 directional pairs), since every prior
    milestone found text the strongest single modality by a wide margin.
    See docs/superpowers/specs/2026-08-06-milestone4-crossmodal-dialogue-design.md
    for the full reasoning.

TERMS YOU'LL SEE
    - cross-attention : one sequence (the "query") looks at another
      sequence (the "key"/"value") to decide what to pay attention to --
      unlike self-attention, where a sequence looks at itself
    - query / key / value : the three inputs attention mechanisms use --
      informally, "what am I looking for" (query), "what's available to
      match against" (key), and "what do I actually retrieve" (value)
    - dialogue window : the current (target) utterance plus up to 4
      utterances immediately before it in the same conversation
    - padding mask : marks positions in a batch that aren't real data
      (e.g. an episode's first utterance has no 4th/3rd/2nd/1st prior
      utterance to fill a full window) so attention ignores them instead
      of treating placeholder zeros as real content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE = 4
WINDOW_SIZE = CONTEXT_SIZE + 1  # +1 for the target utterance itself


class CrossModalDialogueModel(nn.Module):
    def __init__(self, text_dim, audio_dim=768, video_dim=512, hidden_dim=128,
                 num_classes=20, use_cross_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.audio_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.video_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        dialogue_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=2 * hidden_dim, batch_first=True
        )
        self.dialogue_self_attn = nn.TransformerEncoder(dialogue_layer, num_layers=1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_utterance(self, text_feat, audio_seq, video_seq):
        """
        text_feat:  (batch, text_dim)
        audio_seq:  (batch, 8, audio_dim)
        video_seq:  (batch, 5, video_dim)
        Returns: (batch, hidden_dim) -- one enriched representation per utterance.
        """
        text_repr = self.text_proj(text_feat).unsqueeze(1)  # (batch, 1, H)
        audio_repr = self.audio_proj(audio_seq)              # (batch, 8, H)
        video_repr = self.video_proj(video_seq)               # (batch, 5, H)

        if self.use_cross_attention:
            audio_enriched, _ = self.audio_to_text_attn(text_repr, audio_repr, audio_repr)  # (batch, 1, H)
            video_enriched, _ = self.video_to_text_attn(text_repr, video_repr, video_repr)  # (batch, 1, H)
        else:
            # Ablation: same information available to the model, but combined
            # via plain mean-pooling instead of cross-attention -- isolates
            # what cross-attention specifically contributes, on top of the
            # same encoders/dialogue self-attention/classifier.
            audio_enriched = audio_repr.mean(dim=1, keepdim=True)
            video_enriched = video_repr.mean(dim=1, keepdim=True)

        enriched = text_repr + audio_enriched + video_enriched  # (batch, 1, H)
        return enriched.squeeze(1)  # (batch, H)

    def forward(self, text_window, audio_window, video_window, padding_mask=None):
        """
        text_window:  (batch, WINDOW_SIZE, text_dim)
        audio_window: (batch, WINDOW_SIZE, 8, audio_dim)
        video_window: (batch, WINDOW_SIZE, 5, video_dim)
        padding_mask: (batch, WINDOW_SIZE) bool, True = padding position to
                       ignore. The target position (last) is never padding.
                       None means no padding (every position is real).
        (Positions 0..WINDOW_SIZE-2 are context, oldest-first; position
        WINDOW_SIZE-1 is the target utterance.)
        Returns: logits (batch, num_classes)
        """
        batch, window, _ = text_window.shape
        utterance_reprs = []
        for pos in range(window):
            repr_pos = self.encode_utterance(
                text_window[:, pos, :], audio_window[:, pos, :, :], video_window[:, pos, :, :]
            )
            utterance_reprs.append(repr_pos)
        sequence = torch.stack(utterance_reprs, dim=1)  # (batch, window, H)

        fused_sequence = self.dialogue_self_attn(sequence, src_key_padding_mask=padding_mask)  # (batch, window, H)
        target_repr = fused_sequence[:, -1, :]  # last position = target utterance

        logits = self.classifier(target_repr)
        return logits


if __name__ == "__main__":
    # Smoke test: run with `python src/crossmodal_model.py`
    torch.manual_seed(0)
    batch, text_dim, num_classes = 4, 1400, 20
    model = CrossModalDialogueModel(text_dim=text_dim, num_classes=num_classes)

    text_window = torch.randn(batch, WINDOW_SIZE, text_dim)
    audio_window = torch.randn(batch, WINDOW_SIZE, 8, 768)
    video_window = torch.randn(batch, WINDOW_SIZE, 5, 512)
    labels = torch.randint(0, num_classes, (batch,))

    logits = model(text_window, audio_window, video_window)
    assert logits.shape == (batch, num_classes), f"unexpected logits shape {logits.shape}"
    loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss), "loss is not finite"

    # Also smoke-test the padding mask path and the no-cross-attention ablation.
    padding_mask = torch.zeros(batch, WINDOW_SIZE, dtype=torch.bool)
    padding_mask[:, :2] = True  # first 2 context positions are padding for this batch
    logits_padded = model(text_window, audio_window, video_window, padding_mask=padding_mask)
    assert logits_padded.shape == (batch, num_classes), "padding-mask path shape mismatch"
    assert torch.isfinite(logits_padded).all(), "padding-mask path produced non-finite logits"

    ablation_model = CrossModalDialogueModel(text_dim=text_dim, num_classes=num_classes, use_cross_attention=False)
    logits_ablation = ablation_model(text_window, audio_window, video_window)
    assert logits_ablation.shape == (batch, num_classes), "ablation model shape mismatch"

    print("Smoke test passed.")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"padding-mask path: OK, ablation (use_cross_attention=False) path: OK")
