"""GPT-style decoder implemented in PyTorch; no pretrained model or weights."""
import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

LEGACY_MODEL_FIELDS = ('vocab_size', 'context', 'width', 'layers', 'heads', 'dropout')


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 256
    width: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    architecture: str = 'gpt'
    kv_heads: int | None = None
    ffn_width: int | None = None
    rope_theta: float = 10000.0

    def __post_init__(self):
        dimensions = (self.vocab_size, self.context, self.width, self.layers, self.heads)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("width must divide into heads and dropout must be in [0,1)")
        if self.architecture not in {'gpt', 'modern'}:
            raise ValueError('architecture must be gpt or modern')
        if self.architecture == 'modern':
            self.kv_heads = self.heads if self.kv_heads is None else self.kv_heads
            self.ffn_width = math.ceil(8 * self.width / 3 / 128) * 128 if self.ffn_width is None else self.ffn_width
            if (type(self.kv_heads) is not int or type(self.ffn_width) is not int
                    or self.kv_heads < 1 or self.heads % self.kv_heads or self.ffn_width < 1):
                raise ValueError('Attention heads must divide into KV heads; FFN width must be positive')
            if (self.width // self.heads) % 2:
                raise ValueError('RoPE requires an even head dimension')
        elif self.kv_heads is not None or self.ffn_width is not None:
            raise ValueError('KV heads and SwiGLU width require the modern architecture')
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError('RoPE theta must be finite and positive')

    def to_dict(self):
        data = asdict(self)
        # Keep the exact historical schema for existing checkpoints/resume plans.
        return data if self.architecture == 'modern' else {key: data[key] for key in LEGACY_MODEL_FIELDS}


def model_config(vocab_size, settings):
    return ModelConfig(vocab_size=vocab_size, **{field.name: settings[field.name] for field in fields(ModelConfig)
                                               if field.name != 'vocab_size' and field.name in settings})


def inherit_architecture(settings, saved):
    """Copy all base dimensions, preserving the shape of legacy training configs."""
    base = ModelConfig(**saved)
    return dict(settings, **{field.name: getattr(base, field.name) for field in fields(ModelConfig)
                            if field.name != 'vocab_size' and (field.name in saved or field.name in settings)})


def parameter_count(cfg):
    w = cfg.width
    if cfg.architecture == 'gpt':
        return (cfg.vocab_size + cfg.context) * w + cfg.layers * (12 * w**2 + 9 * w) + 2 * w
    kv = cfg.kv_heads * (w // cfg.heads)
    return cfg.vocab_size * w + cfg.layers * (2 * w**2 + 2 * w * kv + 3 * w * cfg.ffn_width + 2 * w) + w


class Rotary(nn.Module):
    def __init__(self, dimension, theta):
        super().__init__()
        self.register_buffer('frequencies', theta ** (-torch.arange(0, dimension, 2).float() / dimension), persistent=False)

    def forward(self, q, k, offset=0):
        positions = torch.arange(offset, offset + q.shape[-2], device=q.device, dtype=torch.float32)
        angles = positions[:, None] * self.frequencies[None, :]
        cosine, sine = angles.cos().to(q.dtype)[None, None], angles.sin().to(q.dtype)[None, None]
        def rotate(x):
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)
        return rotate(q), rotate(k)


class GroupedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.dimension, self.dropout = cfg.width // cfg.heads, cfg.dropout
        self.query = nn.Linear(cfg.width, cfg.width, bias=False)
        self.key_value = nn.Linear(cfg.width, 2 * cfg.kv_heads * self.dimension, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)
        self.rope = Rotary(self.dimension, cfg.rope_theta)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None, use_cache=False):
        b, t, _ = x.shape
        q = self.query(x).view(b, t, self.heads, self.dimension).transpose(1, 2)
        k, v = self.key_value(x).chunk(2, dim=-1)
        k, v = (tensor.reshape(b, t, self.kv_heads, self.dimension).transpose(1, 2) for tensor in (k, v))
        offset = past[0].shape[-2] if past is not None else 0
        q, k = self.rope(q, k, offset)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=-2), torch.cat((past[1], v), dim=-2)
        cache = (k, v) if use_cache else None
        # Explicit expansion also works on Pascal's SDPA math backend.
        groups = self.heads // self.kv_heads
        if groups > 1:
            k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        mask = None
        if offset and t > 1:
            mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= torch.arange(offset, offset + t, device=x.device)[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=not offset,
                                          dropout_p=self.dropout if self.training else 0.0)
        y = self.resid_dropout(self.proj(y.transpose(1, 2).contiguous().view(b, t, -1)))
        return (y, cache) if use_cache else y


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.width, cfg.ffn_width, bias=False)
        self.up = nn.Linear(cfg.width, cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.width, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.RMSNorm(cfg.width, eps=1e-6), nn.RMSNorm(cfg.width, eps=1e-6)
        self.attn, self.mlp = GroupedAttention(cfg), SwiGLU(cfg)

    def forward(self, x, past=None, use_cache=False):
        result = self.attn(self.ln1(x), past, use_cache)
        x = x + (result[0] if use_cache else result)
        x = x + self.mlp(self.ln2(x))
        return (x, result[1]) if use_cache else x


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (a.view(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                 nn.Linear(4 * cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.token = nn.Embedding(cfg.vocab_size, cfg.width)
        modern = cfg.architecture == 'modern'
        self.position = None if modern else nn.Embedding(cfg.context, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*((ModernBlock if modern else Block)(cfg) for _ in range(cfg.layers)))
        self.norm = nn.RMSNorm(cfg.width, eps=1e-6) if modern else nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.token.weight
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_((block.mlp.down if modern else block.mlp[2]).weight, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, *, label_smoothing=0.0):
        t = ids.shape[1]
        if not 0 < t <= self.cfg.context:
            raise ValueError("input length must be within model context")
        x = self.token(ids)
        if self.position is not None:
            x = x + self.position(torch.arange(t, device=ids.device))
        x = self.dropout(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.gradient_checkpointing and self.training and torch.is_grad_enabled() else block(x)
        x = self.norm(x)
        # Only the last position is needed when sampling.
        logits = self.head(x if targets is not None else x[:, -1:, :])
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1),
                                                          label_smoothing=label_smoothing)
        return logits, loss

    @torch.inference_mode()
    def forward_cached(self, ids, cache=None):
        if self.training or self.cfg.architecture != 'modern':
            raise ValueError('KV caching requires a modern model in evaluation mode')
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError('Cache layer count differs from the model')
        offset = cache[0][0].shape[-2] if cache else 0
        if not 0 < ids.shape[1] or offset + ids.shape[1] > self.cfg.context:
            raise ValueError('Cached input exceeds the model context')
        x, updated = self.dropout(self.token(ids)), []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(x, cache[index] if cache else None, True)
            updated.append(layer_cache)
        return self.head(self.norm(x[:, -1:, :])), updated

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens=100, temperature=0.8, top_k=40, eos_id=None, *,
                 top_p=1.0, repetition_penalty=1.0, forbidden_ids=(), use_cache=True, collect_stats=False):
        if self.training:
            raise RuntimeError("call eval() before generation")
        if (max_new_tokens < 0 or temperature < 0 or not math.isfinite(temperature) or top_k < 0
                or not 0 < top_p <= 1 or not math.isfinite(repetition_penalty) or repetition_penalty < 1):
            raise ValueError("invalid generation options")
        prompt_length = ids.shape[1]
        cache, entropy_total, probability_total, generated, stopped = None, 0.0, 0.0, 0, False
        cached = use_cache and getattr(self.cfg, 'architecture', 'gpt') == 'modern'
        for _ in range(max_new_tokens):
            if cached:
                # Rebuild when sliding the window so cached and uncached decoding agree.
                if cache is None or cache[0][0].shape[-2] >= self.cfg.context:
                    logits, cache = self.forward_cached(ids[:, -self.cfg.context:])
                else:
                    logits, cache = self.forward_cached(ids[:, -1:], cache)
            else:
                logits, _ = self(ids[:, -self.cfg.context:])
            logits = logits[:, -1, :].float()
            if collect_stats:
                raw_log_probs = logits.log_softmax(-1)
                raw_probs = raw_log_probs.exp()
                entropy_total = entropy_total - (raw_probs * raw_log_probs).sum(-1).mean()
            if repetition_penalty != 1 and ids.shape[1] > prompt_length:
                # Penalize generated tokens only, in a bounded recent window.
                seen = ids[:, max(prompt_length, ids.shape[1] - 128):]
                scores = logits.gather(1, seen)
                scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                logits = logits.scatter(1, seen, scores)
            if forbidden_ids:
                logits[:, list(forbidden_ids)] = float('-inf')
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1:
                    sorted_logits, indices = logits.sort(descending=True, dim=-1)
                    remove = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits = logits.scatter(1, indices, sorted_logits.masked_fill(remove, float('-inf')))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            generated += 1
            if collect_stats:
                probability_total = probability_total + raw_probs.gather(1, next_id).mean()
            if eos_id is not None and (next_id == eos_id).all():
                stopped = True
                break
        if collect_stats:
            return ids, dict(mean_token_entropy_nats=float(entropy_total / generated) if generated else None,
                             mean_selected_token_probability=float(probability_total / generated) if generated else None,
                             stop_reason='eos' if stopped else 'token_limit', probability_is_factual_confidence=False,
                             cached_decoding=cached)
        return ids

    def config_dict(self):
        return self.cfg.to_dict()
