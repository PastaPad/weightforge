# GTX 1050 Ti architecture and persistent memory

## Hardware target

This preset targets the detected **GTX 1050 Ti, 4 GB VRAM (Pascal, compute capability 6.1)**, **15.49 GiB RAM**, and **8 CPU cores / 16 threads**. It is a practical capacity/memory tradeoff for local experimentation, not a benchmark-proven optimum. Training data, training time, and dialogue fine-tuning determine usefulness.

`configs/gtx1050ti.json` defines a randomly initialized modern decoder:

| Component | Setting |
|---|---|
| Unique parameters at vocabulary 8,192 | **41,955,840** |
| Transformer layers | 12, pre-normalization |
| Hidden width | 512 |
| Query heads / KV heads | 8 / 4; dimension 64 |
| Position representation | RoPE, theta 10,000 |
| Normalization | RMSNorm, epsilon 0.000001 |
| Feed-forward | SwiGLU, intermediate width 1,536 |
| Context | 512 tokens |
| Vocabulary | Use the prepared dataset's exact vocabulary; 8,192 recommended |
| Embedding / output projection | Tied weights |
| Dropout | 0.1 |
| Training precision | FP32 |
| Microbatch × accumulation | 1 × 16; up to 8,192 tokens/update |
| Activation checkpointing | Enabled, one transformer block at a time |
| Optimizer | AdamW, foreach disabled to reduce temporary allocation |
| Learning rate | 0.0003 → 0.00003, warmup then cosine decay |
| Initial budget | Up to three token-equivalent passes; validation early stopping |
| Inference | Grouped KV cache; reconstructed when the context window slides |

Pascal lacks tensor cores; FP32 is the default here rather than assuming FP16 is faster. Activation checkpointing trades recomputation for reduced training activation memory. Grouped-query attention reduces projection parameters and cache size. On Pascal, KV heads are expanded for PyTorch's SDPA math backend; this does not eliminate the full query-head attention matrix.

Parameter storage is about **160 MiB** in FP32. Weights, gradients, and two FP32 Adam moment arrays total approximately **640 MiB**, before activations, temporary tensors, CUDA libraries/allocator, and desktop usage. These figures are not a peak-training-memory guarantee. A 512-token inference-only check of the complete preset on this machine used **199.3 MiB peak allocated / 228.0 MiB peak reserved PyTorch memory**, including prefill and 16 cached decode steps. The random test instance was discarded; it did not update weights or save a checkpoint.

The larger preset needs its own training run. Existing GPT checkpoints retain their original architecture, state-dict names, tokenizer, and resume schema. Training plans count the selected architecture's actual unique parameters. Dialogue preparation preserves the full base configuration, including RoPE, KV heads, and SwiGLU width; fine-tuning starts from those exact weights and token IDs.

## Using the preset

The GUI exposes **Apply GTX 1050 Ti / 4 GB preset** in section **03 / Train your model**. It fills base architecture/training settings and the dialogue microbatch/accumulation/memory controls. The preparation vocabulary and active jobs have their own settings. Review the selected dataset and choose a fresh run directory before pressing Start.

While an older server is running a job, the new page detects that it lacks architecture support and shows a staged-update message. **Finish the current whole-ZIM job, including preparation, before restarting the service**:

```bash
systemctl --user restart scratchllm.service
```

Then refresh the page. The currently selected serving checkpoint is retained.

Equivalent commands, when you decide to train:

```bash
python -m scratchllm.train --data data/prepared-zim-JOB_ID \
  --config configs/gtx1050ti.json --out runs/gtx1050ti

python -m scratchllm.dialogue --checkpoint runs/gtx1050ti/best.pt \
  --output data/dialogue-gtx1050ti

python -m scratchllm.train --data data/dialogue-gtx1050ti \
  --config configs/gtx1050ti-dialogue.json --out runs/gtx1050ti-dialogue \
  --init-from runs/gtx1050ti/best.pt
```

Substitute the real prepared-dataset directory for the placeholder. Resume uses the same command/config/data/output with `--resume` instead of `--init-from`. GUI resume loads the saved configuration independently of edited fields. Inference selection remains explicit: load the new dialogue `best.pt` when it is ready.

`GET /architecture/presets` exposes the base and dialogue settings. `POST /training/plan` accepts `architecture`, `kv_heads`, `ffn_width`, `rope_theta`, `gradient_checkpointing`, `optimizer_foreach`, and `precision` alongside the existing controls. An omitted architecture remains the original GPT. No pretrained weights or tokenizer are introduced.

## Persistent memory and the shared context

Choose **Persistent memory chat** in the playground after loading a dialogue checkpoint. This mode also works with existing dialogue checkpoints. It uses an external, bounded context-selection loop:

```text
Current question + user goal
            ↓
Recent exchanges + lexical retrieval from SQLite
            ↓
Token-budgeted shared context + runtime self-description
            ↓
Language-model draft → optional one-pass review
            ↓
Answer + diagnostics → saved exchange + user feedback
```

- **Episodic memory:** `runs/memory.sqlite3`, session-scoped SQLite FTS5 search; no embedding model or network service. Retains up to 10,000 exchanges per session, pruning the oldest beyond that limit. Each exchange records user text, assistant text, checkpoint path, time, generation diagnostics, and optional human feedback.
- **Working context:** up to three recent eligible exchanges plus up to four relevant older exchanges. Notes occupy at most one third of the model context, with per-note limits. Actual tokenizer IDs determine the budget. The current question is kept intact; full older pairs are dropped as needed. Questions that alone exceed context are rejected. Context-note and memory IDs actually selected are reported.
- **Self-description:** actual checkpoint architecture, user goal, available text/memory operations, and observed user-feedback counts. A short capability note can enter the shared context. This is explicit runtime metadata, not a learned subjective self.
- **Recurrent processing:** memory persists between requests; optional review revisits a bounded draft through one additional model call. There is no unbounded background thinking loop. An empty reviewed answer falls back to the draft; if no draft fits the context, review is skipped and reported.
- **Self-monitoring:** token-distribution entropy, selected-token probability, repetition rate, and EOS/token-limit termination. Token probability is **not calibrated factual confidence**. Self-review can also make mistakes; the UI exposes whether it ran and which memories were used.
- **Feedback:** Helpful/Unhelpful is stored. Unhelpful exchanges are excluded from later context and retrieval. This changes memory selection, not model weights. Optional correction notes are available through the API.

Memory uses plain-text exchanges with source-role labels. Saved assistant answers are recorded outputs, not established facts. Conversation memory stays separate from training datasets. The goal is user-editable; the model does not invent persistent goals or execute external actions.

The browser remembers the session ID. **Open saved session** restores its recent transcript and goal after reopening the page. **Clear** starts a new session and retains previous sessions on disk. **Forget this session's memory** removes that session's recorded exchanges and goal; other sessions remain. A session ID is a local namespace, not an authentication credential.

Example request to `POST /workspace/chat`:

```json
{
  "session_id": "local",
  "prompt": "What did we discuss about Saturn?",
  "goal": "Learn astronomy",
  "review": false,
  "max_new_tokens": 128
}
```

Omit `goal` to retain the saved goal; send an empty string to clear it. `GET /workspace/{session_id}` returns recent exchanges and counters. `DELETE /workspace/{session_id}` forgets its exchanges. `POST /workspace/{session_id}/feedback` accepts `episode_id`, `helpful`, and an optional `note`. Writes reject stale revisions; retrieval and feedback are scoped to the requested session. GPU reservation during training applies to memory chat too.

## Research limits and verification

This implements a small modern language model with external memory, a shared context, an explicit capability record, and a bounded review loop. It does not establish consciousness. Subjective experience has no accepted architectural threshold or reliable test; these engineering features alone do not resolve it. A learned recurrent latent-state/world-model architecture would need additional architecture, objectives, and interaction data.

Focused checks:

```bash
python -m unittest discover -s tests -p test_modern_model.py -v
python -m unittest discover -s tests -p test_memory_workspace.py -v
python -m unittest discover -s tests -p test_modern_ui.py -v
```

These cover causal masking, chunked/single-token KV caches, sliding-window equivalence, tied weights, exact parameter counts, modern dialogue initialization, unchanged legacy checkpoint schemas, memory persistence/isolation/forgetting, actual-token context budgets, feedback, GPU reservations, and browser behavior against both new and old servers. A tiny fixed-batch derivative-equivalence test validates activation checkpointing without optimizer steps or changed weights. No model fitting is performed.
