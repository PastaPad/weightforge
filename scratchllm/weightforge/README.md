# Weight-space modeling (`scratchllm.weightforge`)

Treats model weights as data: measure, edit, and eventually predict them.

The long-term goal is a model that reads the weights of open English LLMs and
generates weights for a requested architecture/task, with no gradient training.
The current stage is empirical: prove that weight-space structure is measurable
and predictable on real open checkpoints.

## What is implemented

| Piece | Module | Status |
|---|---|---|
| scratchllm checkpoint loading + arithmetic | `corpus.py`, `functional.py` | done |
| HF safetensors loading | `corpus.py` (`load_hf_checkpoint`) | done |
| Minimal Qwen2 forward pass (no `transformers`) | `hf_models.py` | done, verified by generation |
| Next-token NLL for HF models | `functional.py` (`evaluate_hf_nll`) | done |
| Interpolation / task-vector / probe sweep | `hf_experiment.py` (`run_pair`) | done |
| Pair connectivity diagnostics | `hf_experiment.py` (`weight_similarity`) | done |
| Corpus runner over a pair spec | `hf_experiment.py` (`run_corpus`) | done |
| Synthetic task corpus (scratchllm) | `build.py`, `tasks.py` | done, not the research corpus |

## Usage

Corpus spec: `configs/weightforge/hf-corpus.json` (evaluation stream + list of
base/tuned pairs). Per-pair results land in the output directory as JSON.

```bash
# one pair
python -m scratchllm.weightforge hf-smoke \
  --base-id Qwen/Qwen2.5-0.5B \
  --tuned-id Qwen/Qwen2.5-0.5B-Instruct \
  --val /tmp/opencode/wf-eval/qwen-gutenberg/val.bin \
  --out runs/weightforge/qwen-smoke/results.json

# all pairs in a spec
python -m scratchllm.weightforge hf-corpus \
  --config configs/weightforge/hf-corpus.json \
  --out-dir runs/weightforge/hf-corpus

# tabulate a finished corpus
python -m scratchllm.weightforge hf-summary --dir runs/weightforge/hf-corpus
```

Evaluation streams are little-endian `uint32` token files (`train.bin`/`val.bin`
convention). The current one is a Qwen-tokenized Project Gutenberg split:
Alice in Wonderland + Sherlock Holmes + Pride and Prejudice, 90/10 split,
34,992 validation tokens.

## Findings so far (Qwen2.5, English Gutenberg + Python stdlib NLL)

Verified against `transformers` on the same weights: logits agree to 2e-5, NLL to
1e-6.

`Qwen2.5-0.5B` -> `Qwen2.5-0.5B-Instruct` (median weight cosine 0.999, `sibling`):

| alpha / scale | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| interpolation, english | 2.996 | 3.002 | 3.014 | 3.033 | 3.060 |
| interpolation, code | 1.424 | 1.424 | 1.428 | 1.436 | 1.450 |
| task vector, english | 2.996 | 3.014 | 3.060 | 3.175 (x1.5) | 3.544 (x2.0) |
| task vector, code | 1.424 | 1.428 | 1.450 | 1.497 (x1.5) | 1.586 (x2.0) |

Monotone and near-linear; the instruct delta costs a little raw-text likelihood
and costs a little code likelihood too.

`Qwen2.5-Coder-0.5B` -> `Qwen2.5-Coder-0.5B-Instruct` (median cosine 0.784,
`continued-pretraining`):

| alpha / scale | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| interpolation, english | 3.050 | 3.939 | 5.076 | 4.162 | 3.712 |
| interpolation, code | 1.341 | 1.857 | 2.491 | 1.764 | 1.450 |
| task vector, english | 3.050 | 5.076 (x0.5) | 3.712 (x1.0) | 15.62 (x1.5) | 25.67 (x2.0) |
| task vector, code | 1.341 | 2.492 (x0.5) | 1.450 (x1.0) | 15.99 (x1.5) | 25.10 (x2.0) |

Same nominal lineage, but the pair is **not linearly connected**: a loss barrier
at the midpoint and divergent extrapolation. Median weight cosine separates the
two cases cleanly (0.999 vs 0.784), which is why every pair now records
connectivity diagnostics.

## Corpus overview

| pair | label | median cos | median relative delta | barrier |
|---|---|---|---|---|
| Qwen2.5-0.5B -> Instruct | sibling | 0.999 | 0.053 | 0.000 |
| Qwen2.5-1.5B -> Instruct | sibling | 1.000 | 0.011 | 0.000 |
| Coder-0.5B -> Coder-0.5B-Instruct | continued-pretraining | 0.784 | 0.685 | +1.364 |

The instruct delta is *relatively* smaller at 1.5B than at 0.5B (0.011 vs 0.053),
and its interpolation path is correspondingly flatter (english 2.594 -> 2.629 vs
2.996 -> 3.060). Barrier height tracks relative delta norm, not nominal lineage:
the coder pair has the same parent/child naming but a 12x larger delta.

## Delta transfer

Some deltas also fail to *compose*. Applying the code-pretraining delta
(`Coder-0.5B - Qwen2.5-0.5B`) to the chat base at half strength lands in a
barrier (english 4.70, code 2.32) even though the full-strength delta reproduces
`Coder-0.5B` exactly. Only sibling-scale deltas stay usable off their endpoints.

| state | english NLL | code NLL |
|---|---|---|
| `Qwen2.5-0.5B` | 2.996 | 1.424 |
| `+ T_code` x1.0 (= `Coder-0.5B`) | 3.050 | 1.341 |
| `+ T_code` x0.5 | 4.703 | 2.324 |
| `+ T_chat` x1.0 (= `Instruct`) | 3.060 | 1.450 |
| `+ T_code + T_chat` | 3.200 | 1.408 |

## Honest limitations

- One 0.5B family, one evaluation corpus, one metric (NLL). No perplexity on
  task-specific data yet (no chat-template scoring, no code benchmark).
- The probe (`ridge LOO R^2`) has 5 points per curve; it is a wiring check, not a
  result.
- CPU evaluation at ~13 s per 4x1024-token window makes 1.5B+ sweeps slow.
- `hf_models.py` supports the Qwen2 architecture only.
