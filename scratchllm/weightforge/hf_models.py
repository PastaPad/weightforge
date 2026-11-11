"""Minimal Qwen2-style decoder in pure PyTorch for weight-space evaluation.

Loads HF safetensors weights and runs next-token NLL on token streams.
No transformers/torchaudio dependency — just PyTorch ops.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 32768

    @staticmethod
    def from_dict(d: dict) -> "Qwen2Config":
        return Qwen2Config(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            intermediate_size=d["intermediate_size"],
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 1000000.0),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            max_position_embeddings=d.get("max_position_embeddings", 32768),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x_normed).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = theta ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin for the given positions.

        Qwen2 uses the NeoX ("half-split") rotary convention: frequencies are
        computed for head_dim/2 pairs and then duplicated so that the first and
        second halves of a head are rotated by the same angle. ``x`` is only used
        for device/dtype, the positions come from ``position_ids``.
        """
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        positions = position_ids.to(torch.float32)
        if positions.dim() == 1:
            positions = positions[None, :]
        freqs = positions.unsqueeze(-1) * inv_freq           # [bsz, seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)              # [bsz, seq, head_dim]
        cos = emb.cos()[:, None]                             # [bsz, 1, seq, head_dim]
        sin = emb.sin()[:, None]
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to q, k. q,k: [bsz, num_heads, seq, head_dim]."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple | None]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        if use_cache:
            present = (key_states, value_states)
        else:
            present = None

        # GQA: repeat K,V for grouped query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=past_key_value is None and q_len > 1,
            dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(hidden_states, position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                past_key_values: list | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, list | None]:
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        next_past = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            hidden_states, present = layer(hidden_states, position_ids, past, use_cache)
            if use_cache:
                next_past.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_past if use_cache else None


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                past_key_values: list | None = None, use_cache: bool = False,
                label_smoothing: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        hidden_states, past = self.model(input_ids, past_key_values, use_cache)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                label_smoothing=label_smoothing
            )
        return logits, loss, past

    @torch.inference_mode()
    def forward_cached(self, input_ids: torch.Tensor, cache: list | None = None):
        hidden_states, new_cache = self.model(input_ids, cache, use_cache=True)
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits, new_cache


def load_qwen2_state(state_dict: dict[str, torch.Tensor], config: Qwen2Config,
                     device: str = "cpu", dtype: torch.dtype | None = None) -> Qwen2ForCausalLM:
    """Build a Qwen2 model and load an HF state dict.

    ``dtype`` defaults to the checkpoint dtype so large checkpoints are not
    silently doubled in memory by an fp32 copy.
    """
    if dtype is None:
        dtype = next(iter(state_dict.values())).dtype
    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)

    # Map HF Qwen2 keys to our module names
    mapped = {}
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace("model.", "")
        if nk.startswith("embed_tokens"):
            nk = "model.embed_tokens.weight"
        elif nk.startswith("layers."):
            parts = nk.split(".")
            layer_idx = int(parts[1])
            if parts[2] == "input_layernorm":
                nk = f"model.layers.{layer_idx}.input_layernorm.weight"
            elif parts[2] == "post_attention_layernorm":
                nk = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            elif parts[2] == "mlp":
                if parts[3] == "gate_proj":
                    nk = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                elif parts[3] == "up_proj":
                    nk = f"model.layers.{layer_idx}.mlp.up_proj.weight"
                elif parts[3] == "down_proj":
                    nk = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            elif parts[2] == "self_attn":
                if parts[3] == "q_proj":
                    nk = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                    if "bias" in parts[4]:
                        nk = f"model.layers.{layer_idx}.self_attn.q_proj.bias"
                elif parts[3] == "k_proj":
                    nk = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                    if "bias" in parts[4]:
                        nk = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
                elif parts[3] == "v_proj":
                    nk = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                    if "bias" in parts[4]:
                        nk = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
                elif parts[3] == "o_proj":
                    nk = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        elif nk == "norm.weight":
            nk = "model.norm.weight"

        mapped[nk] = v

    # Handle tied lm_head
    if "model.embed_tokens.weight" in mapped and "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"]

    model.load_state_dict(mapped, strict=True)
    return model.eval()