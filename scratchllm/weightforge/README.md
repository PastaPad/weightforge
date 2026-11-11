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
| LoRA/PEFT adapters as exact deltas | `adapters.py` | done |
| Delta composition + interaction terms | `compose.py` | done |
| Held-out delta reconstruction from a basis | `lowrank.py` | done |
| Recipe x data variance decomposition | `variance.py` | done |
| Stable float64 reductions for 100M+ tensors | `numerics.py` | done |
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

# additive composition of several deltas, with interaction terms
python -m scratchllm.weightforge hf-compose \
  --base-id Qwen/Qwen2.5-0.5B-Instruct \
  --tuned-id saidutta69/Qwen2.5-0.5B-Instruct-heretic \
  --tuned-id sallani/ISO27001-Qwen2.5-0.5B-Edge \
  --eval english=/tmp/opencode/wf-eval/qwen-gutenberg/val.bin \
  --out runs/weightforge/compose/example.json

# can a held-out delta be rebuilt from a basis of other deltas?
python -m scratchllm.weightforge hf-lowrank \
  --base-id Qwen/Qwen2.5-0.5B-Instruct \
  --tuned-id <model-a> --tuned-id <model-b> --tuned-id <model-c> \
  --eval english=/tmp/opencode/wf-eval/qwen-gutenberg/val.bin

# recipe x data decomposition of a controlled family (adapters supported)
python -m scratchllm.weightforge hf-variance \
  --base-id Qwen/Qwen2.5-0.5B-Instruct \
  --cell r1,dataA,<adapter-a> --cell r1,dataB,<adapter-b> \
  --cell r32,dataA,<adapter-c> --cell r32,dataB,<adapter-d>
```

Evaluation streams are little-endian `uint32` token files (`train.bin`/`val.bin`
convention). The current one is a Qwen-tokenized Project Gutenberg split:
Alice in Wonderland + Sherlock Holmes + Pride and Prejudice, 90/10 split,
34,992 validation tokens.

## Findings so far

Probes are English Project Gutenberg (Alice, Sherlock, Pride and Prejudice) and
Python 3.14 stdlib source, scored as next-token NLL on four 1024-token windows.
Hand-written forward pass verified against `transformers`: logits agree to 2e-5,
NLL to 1e-6.

### Corpus

| pair | label | median cos | total rel delta | changed tensors | barrier | english endpoint |
|---|---|---|---|---|---|---|
| 0.5B -> Instruct | sibling | 0.999 | 0.042 | 100% | 0.000 | 3.061 |
| 1.5B -> 1.5B Instruct | sibling | 1.000 | 0.010 | 100% | 0.000 | 2.629 |
| Instruct -> heretic | sibling | 1.000 | 0.008 | 9.7% | 0.000 | 3.074 |
| Instruct -> ISO27001 | sibling | 1.000 | 0.081 | 38.6% | 0.000 | 4.633 |
| Instruct -> ISO42001 | sibling | 1.000 | 0.043 | 38.6% | 0.000 | 4.102 |
| Coder-0.5B -> Coder Instruct | continued-pretraining | 0.784 | 0.501 | 100% | +1.364 | 3.712 |

Two failure modes have to be scored separately:

* **barrier** - the midpoint is worse than both endpoints (non-monotone path);
* **endpoint damage** - the tuned model itself is worse on the probe.

They do not coincide. ISO42001 and the chat delta have almost the same total
relative delta (0.043 vs 0.042) but cost 1.05 and 0.01 NLL. Barrier needs both a
large delta and the wrong direction (only the coder pair has one). Magnitude
alone predicts neither.

The first two rows start from `Qwen2.5-0.5B`/`1.5B` and the rest from
`-Instruct`, so absolute NLLs are not comparable across rows; the deltas are.

`median_cosine` is useless for surgical edits: most tensors are bit-identical, so
the median is exactly 1.0 while 9.7% of tensors moved. Every pair now records
`total_relative_delta` and `changed_fraction` alongside it.

### Surgical edits are common and behave like siblings

`saidutta69/Qwen2.5-0.5B-Instruct-heretic` changes exactly `self_attn.o_proj` and
`mlp.down_proj` in every layer (33 of 338 tensors, total relative delta 0.008).
The interpolation path is flat (3.051 -> 3.074 english) and extrapolation is
stable (x2: 3.164 english, 1.520 code). The ISO "Edge" models are LoRA merges
that touch 38.6% of tensors with deltas 5-10x larger; their path is also
monotone, but every intermediate model is already damaged.

### Orthogonal deltas still interfere

Three deltas on the same base, each with a near-zero pairwise cosine:

| state | english | code |
|---|---|---|
| Instruct (base) | 3.051 | 1.445 |
| + heretic | 3.074 | 1.456 |
| + ISO27001 | 4.633 | 2.572 |
| + ISO42001 | 4.102 | 2.195 |
| + all three | 7.289 | 4.684 |

Additive prediction would be 5.71; the measured combination is 7.29, an
interaction of +1.58. Cosine between deltas does not predict whether the losses
add: heretic/ISO cosines are 0.000 and ISO27001/ISO42001 is 0.198, yet the
combination is far past either delta alone.

### Transfer across bases

Applying the code-pretraining delta (`Coder-0.5B - Qwen2.5-0.5B`) to the chat
base at half strength lands in a barrier even though the full-strength delta
reproduces `Coder-0.5B` exactly:

| state | english | code |
|---|---|---|
| `Qwen2.5-0.5B` | 2.996 | 1.424 |
| `+ T_code` x1.0 (= `Coder-0.5B`) | 3.050 | 1.341 |
| `+ T_code` x0.5 | 4.703 | 2.324 |
| `+ T_chat` x1.0 (= `Instruct`) | 3.060 | 1.450 |
| `+ T_code + T_chat` | 3.200 | 1.408 |

A delta is only usable off its endpoints when its magnitude is sibling-scale
(total relative delta below ~0.1 in this family).

### A basis of task vectors reconstructs narrow deltas only

Controlled family published by `gumperto`: five datasets times two LoRA recipes
of the same base. Deltas are exact (adapters, not merged files).

| recipe | support | rank per delta | within-family cos | leave-one-out cos | variance in one direction |
|---|---|---|---|---|---|
| `down-l12-r1` | 1 matrix | 1 | 0.58 - 0.93 (0.71) | 0.75 - 0.95 | 77% |
| `all-full-r32` | 168 matrices | 32 | 0.03 - 0.33 (0.09) | 0.09 - 0.36 | 32% |

For the rank-1 recipe a single shared direction explains 77% of the family's
total delta variance (90% with two directions), and a held-out dataset's delta is
recovered with cosine 0.75-0.95 from the other four. For the rank-32 recipe the
spectrum is nearly flat (32/21/18/14/12%) and leave-one-out recovery is poor.

Same data, different recipe: |cos| < 0.001. Different data, same recipe: strongly
aligned for the rank-1 recipe, barely aligned for the rank-32 recipe.

So a held-out delta is predictable from other deltas of the same recipe exactly
to the extent that the recipe is narrow. Width buys behaviour but destroys linear
predictability, and recipe identity dominates support: the same data trained two
ways lands in nearly orthogonal subspaces.

The reconstruction experiment also shows why NLL is the wrong primary metric
here: projecting a held-out delta onto a basis can *improve* NLL relative to the
real tuned model, because the projection keeps the harmless component and drops
the damaging one. Delta-space fidelity (cosine, norm ratio) is the honest measure.

### Recipe vs data

Two-way decomposition of the ten-delta family:

| decomposition | recipe | data | interaction |
|---|---|---|---|
| by norm | 52% | 24% | 24% |
| scale-normalized (direction) | 35% | 32% | 32% |

Direction is roughly one-third recipe, one-third dataset, one-third interaction.
There is no clean "task direction" that can be lifted from one recipe and reused
in another.

### Adapters barely move these probes

The `gumperto` adapters change english NLL by at most 0.008 and code NLL by at
most 0.003. They were trained on synthetic data outside both probes, so their
delta-space structure is measurable but their behaviour is not. Behavioural
claims about a delta need an in-domain probe; weight-space claims do not.

## Honest limitations

- One model family (Qwen2, 0.5B/1.5B), two English probes, one behavioural metric
  (NLL). No chat-template scoring and no task benchmark yet.
- Delta families have five members; leave-one-out uses a four-vector basis. That
  is enough to separate "narrow" from "wide" recipes, not enough for scaling laws.
- The probe (`ridge LOO R^2`) has five points per curve; it is a wiring check.
- CPU evaluation at ~1-3 minutes per 4x1024-token window makes 1.5B+ sweeps slow
  and limits how many pairs can be measured per session.
- Float32 accumulation silently corrupted large-tensor diagnostics (an embedding
  cosine of 1.033 between identical tensors); `numerics.py` now uses chunked
  float64 for every reduction over 100M+ elements.
- `hf_models.py` supports the Qwen2 architecture only.
