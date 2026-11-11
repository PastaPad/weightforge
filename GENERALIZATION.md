# Generalization repair and research notes

## Human dialogue stage with the existing vocabulary

The user subsequently authorized established dialogue datasets. The optional supervised stage now uses English human-only OpenAssistant OASST1 conversation paths and Dolly 15k instructions/responses, from pinned/checksummed exports. It copies the exact base-checkpoint tokenizer JSON and initializes from its weights. Vocabulary/token IDs and architecture are preserved; weights, including embeddings, can continue learning.

Default fine-tuning is 3 shuffled example passes at LR 0.00005 (minimum 0.000005), microbatch 2 × accumulation 8, and early-stop patience 3. Only the final assistant reply/EOS contributes to loss; prior turns and questions provide context. Validation groups whole conversations/reference passages and reports assistant-only NLL. The GUI can append preparation/fine-tuning to the base pipeline and uses the shared chat template when a dialogue checkpoint is loaded. These are functionally tested starting settings; no new model training was performed to establish output quality.

Tokenizer reuse preserves segmentation, not a hard limit on which words or facts the model can learn. To inherit Wikipedia-trained representations, select a base checkpoint actually trained with Wikipedia text. The earlier selected `runs/kikeline1` checkpoint was board-only. See [DIALOGUE.md](DIALOGUE.md) for source details and commands.

## 2026-10-21: URL-free data, Wikipedia, and response-oriented settings

The current selected run, `runs/kikeline1`, was trained on `data/kikeline3`: 1,699,367 training tokens from `soyjak.st` and `swedishwin.com`, with a 12-layer, width-512 model. Its best logged validation NLL was 5.0708 at step 1900. At step 2400, training NLL was 3.3491 and validation NLL was 5.1068; early stopping ended the run. This is evidence of overfitting, not a reason to increase capacity or repeat the same posts indefinitely.

The requested Wikipedia source now adds English article prose. URL stripping applies before tokenization and to displayed generated completions. A fresh preparation of existing v4 raw text removes URLs; new runs require this text-policy metadata.

`configs/response.json` and new GUI/API defaults keep the earlier 6-layer, width-256 capacity, double context to 512, use microbatch 2 × accumulation 8 (still 8,192 nominal tokens/update), and use shuffled sampling with an adjustable 25-pass budget. Dropout 0.1, LR 0.0003, fixed validation, early-stop patience 6, and best-checkpoint selection remain the starting values. Generation defaults use temperature 0.7, top-p 0.9, top-k 40, and repetition penalty 1.1. These changes have functional tests; their effect on future model quality has **not** been measured by training a new model.

These settings cannot produce early-ChatGPT capability by themselves. The selected raw sources teach their prose and forum styles; assistant instruction-following additionally needs suitable human question/answer examples. No outside corpus or pretrained weights are introduced. The sections below record earlier experiments; their old dataset-training commands predate the new URL-removal requirement.

## Correction after the coherence regression

The small-model preset was promoted before measuring its output quality. That recommendation was too aggressive: it simultaneously reduced model capacity and exposure, doubled dropout, and enabled label smoothing. **A smaller train/validation gap is not evidence of better language generation.**

The user subsequently trained `runs/web5` on `data/prepared-text7`. It contains exactly the same 79 training and 10 validation documents as `prepared-text6`, but a different tokenizer (10,949 vocabulary entries instead of 8,192). The new run stopped at its 987-update limit, roughly 20 token-equivalent passes, with its best validation NLL at the final step. This is consistent with an overly constrained/undertrained model, rather than evidence that validation-triggered stopping had found its optimum. The combined experiment cannot isolate the effect of each changed setting.

Read-only comparison on the **same 10 held-out articles / 59,839 UTF-8 bytes**:

| Checkpoint | Held-out bits per byte (lower is better) |
|---|---:|
| `runs/web4/best.pt` | **2.3283** |
| `runs/web5/best.pt` | 2.4120 |

Method: each model's own tokenizer; lossless encoding checked; FP32 CPU; each document begins with BOS; 256-token context; sliding windows with stride 128; score each text token once; exclude BOS/EOS targets; total text negative log likelihood divided by `ln(2) × UTF-8 bytes`. Byte normalization permits comparison across tokenizers, unlike the raw token perplexities. The differing tokenizers still give different amounts of byte context within 256 tokens. This likelihood comparison supports the regression diagnosis but is not a direct measure of human-rated coherence.

Seed-42 completions at temperatures 0 and 0.8 for `Soyjak is`, `Soyjak Wiki is`, and `The history of soyjaks` also showed much more fragmentation and blank-line collapse in `web5`. Both models still exhibit repetition and factual confusion.

The served model has been restored to **`runs/web4/best.pt`**. GUI/API defaults and `configs/tiny.json` now restore its baseline architecture and optimization settings: 6 layers, width 256, dropout 0.1, accumulation 8, 2,500 maximum updates, 200 warmup updates, and random-window training. Fixed validation and early stopping remain enabled. The arbitrary 20-pass cap, extra smoothing, EMA, and shuffled sampling are opt-in. `configs/small-corpus.json` is retained as an **experimental configuration that regressed in this observed run**, not the recommended default.

No new training was started during this correction. Existing checkpoints and run configurations were preserved. The following sections retain the original overfitting measurements and describe the available mechanisms.

## What happened in the existing run

Audit of `runs/web4` and `data/prepared-text6` on 2026-10-19:

- 6,895,616 unique model parameters.
- 79 training documents / 106,508 tokens; 10 validation documents / 16,016 tokens.
- All 89 documents are from `soyjakwiki.org`; no exact whitespace-normalized document duplicates cross the split.
- 2,500 updates × 8,192 tokens = 20,480,000 sampled tokens, approximately **192 corpus-equivalent passes**. Sampling was with replacement.
- The old log's lowest validation loss was **5.8857 at step 900**. At step 2,500, training-batch loss was **1.7994**, while validation loss had risen to **7.1119**.

This is strong evidence of memorization/overfitting. The old evaluator also changed its random validation sample at every check, making the curve noisy. Training loss included dropout, while validation did not, so that single training-batch value was not a matched comparison.

### Read-only checkpoint re-evaluation

Using the new `fixed-token-nll-v1` evaluator, FP32 CPU, batch size 4, context 256, full train/validation streams, dropout off, and no label smoothing:

| Existing checkpoint | Step | Train NLL | Validation NLL | Gap |
|---|---:|---:|---:|---:|
| `runs/web4/best.pt` | 900 | 3.5838 | 6.0277 | 2.4439 |
| `runs/web4/last.pt` | 2500 | 1.2952 | 7.0608 | 5.7655 |

Each target was evaluated once (106,507 train / 16,015 validation targets), including partial final windows. The first stream token has no preceding input and is excluded. The tokenizer was checked against the checkpoint. These were **forward-only measurements of existing weights**. No optimizer updates or new training occurred. Differences from historical values reflect the changed evaluation protocol; compare runs using the same tokenizer, dataset, and window plan.

## Implemented changes

1. **Fixed, token-weighted evaluation.** A deterministic set of nonoverlapping target blocks is chosen independently of training RNG. The entire split is evaluated when it fits the configured evaluation budget. Larger splits use a fixed seeded subset without replacement. Partial windows are masked and losses are weighted by actual target counts. Train NLL and validation NLL use the same selected weights, no dropout, and unsmoothed cross-entropy. The separately logged `optimization_loss` includes training regularization.
2. **Early stopping.** The default patience is six checks without at least 0.01 NLL cumulative improvement over the last meaningful improvement. Any new absolute minimum still updates `best.pt`. Patience/reference/best-step state persists through interruption. Resuming an early-stopped run does not bypass the decision.
3. **Explicit budget modes and scheduling.** In fixed-update mode, a positive `max_epochs` optionally caps the schedule. In data-pass mode, `max_epochs` is the requested exposure and the trainer derives the update count from corpus size and effective batch, ignoring a stale fixed step count. The last whole update can exceed the nominal token budget by less than one update. Shuffled partial chunks can use fewer tokens than the estimate; metrics report actual exposure. Warmup and cosine decay follow the effective schedule. The baseline still uses fixed updates with no pass cap; no universal number of passes is imposed.
4. **Shuffled passes.** Every training target is visited once per pass, in shuffled chunks. Chunk boundaries rotate between passes. Final partial chunks are retained, and gradient accumulation weights them by target count. Sampler position, boundaries, and RNG are checkpointed.
5. **Exponential moving average (EMA) of weights.** Optional `ema_decay=0.98` averages successful optimizer updates. Initial averaging is warmed up, and skipped FP16 updates do not advance it. Raw and EMA validation NLLs are both measured; the lower one competes for `best.pt`. `last.pt` always contains raw optimizer-compatible weights plus the separate EMA state. Tied embeddings are averaged once.
6. **Optional label smoothing.** The small-corpus preset uses 0.05 smoothing to reduce overconfident fitting. It can help or hurt held-out NLL and is independently switchable. It is disabled during evaluation; reported perplexity always comes from unsmoothed NLL.
7. **Visible controls and an experimental smaller-model preset.** `configs/small-corpus.json` uses 4 layers, width 128, dropout 0.2, and 2,048 nominal tokens/update. It is no longer the GUI default. A metadata-only preview shows parameter count, corpus size, effective schedule, and validation coverage before starting a job.

Existing prepared data, tokenizers, checkpoint files, and model selection are preserved. Legacy interrupted runs retain their saved configuration, sampling, and random validation protocol when resumed; use a fresh run to enable the new preset. Full new resume state includes the evaluation plan, early-stop state, EMA, sampler, optimizer, and RNG states.

## Using the change

The restored checkpoint is immediately usable in the playground. For a controlled comparison with that checkpoint, manually select the same `data/prepared-text6` dataset/tokenizer, a fresh run directory such as `runs/baseline`, and inspect the training preview. The fixed-update baseline plans up to 2,500 updates with 200 warmup updates and earlier stopping on validation stagnation. The GUI now initially selects the **newest complete prepared dataset**, separately from the served model's dataset. Change one experimental option at a time and inspect completions alongside held-out scores. Training starts only when you click **Start training**.

Equivalent command for a user-started new run:

```bash
python -m scratchllm.train --data data/prepared-text6 \
  --config configs/tiny.json --out runs/baseline
```

Use `best.pt` for inference, and `last.pt` for interruption recovery. The metric monitor identifies raw/EMA selection, matched NLL gap, exposure count, best step, and stopping reason.

### Larger dataset and the unchanged 987 count (2026-10-20)

The new `data/prepared-text8` contains 937 training / 106 validation documents, with 1,587,273 training tokens. `runs/web5` saved `steps: 987`, so fixed-update mode or resuming that run cannot automatically increase its budget. The old cap was `min(fixed steps, pass-derived steps)`: larger data could increase the second term while the fixed first term still limited training.

The new **Data passes (auto-size updates)** mode derives the count directly. At 8,192 tokens/update, this dataset gives 3,876 updates for 20 passes, 9,688 for 50, or 19,376 for 100. At the old smaller 2,048-token effective batch, 20 passes give 15,501 updates. These are examples of exposure, not quality recommendations. Automatic preview and dataset selection were corrected; no training was initiated to verify them.

The controls address avoidable overtraining and evaluation noise. The observed small-model run worsened generation despite a smaller loss gap; a newly trained baseline with the corrected defaults has not been evaluated. For interpretable experiments, keep the split/tokenizer/evaluation budget fixed and vary EMA, smoothing, or model size separately. If collecting more data, use the existing four sources and prepare a fresh dataset; changed tokenization makes raw token perplexities incomparable. Ten validation documents give a limited estimate, and repeated hyperparameter selection can itself overfit that holdout.

## Research basis

- Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), NeurIPS 2023, revised 2025. Repeated tokens have diminishing value and excess parameters can be wasteful in data-constrained regimes. Their large-model scaling results motivate tracking repetition and capacity; they do not prescribe an exact epoch limit for this tiny wiki corpus.
- Morales-Brotons et al., [Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits](https://arxiv.org/abs/2411.18704), TMLR 2024. Empirical support for weight averaging as a generalization/consistency/calibration technique. Its effectiveness with this AdamW language model is tested through raw-versus-EMA validation rather than assumed.
- Pereyra et al., [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548), 2017. Studies confidence penalties and label smoothing, including language modeling. This implementation uses label smoothing, not a separate entropy-penalty objective.

These papers informed implementation choices; their text and datasets are not included in training. All learning inputs remain the selected prepared corpus from the existing sources. There is no established training recipe or validated measurement here for making a model conscious; the measurable objective is held-out next-token prediction quality.

## Verification without training

```bash
python -m unittest discover -s tests -p test_generalization.py -v
python -m unittest discover -s tests -p test_serving_selection.py -v
python -m unittest discover -s tests -p test_training_budget_ui.py -v
```

The checks cover causal isolation, fixed evaluation, target weighting/tails, RNG and weight preservation, shuffled coverage/resume, budget/schedule alignment, early stopping, EMA/tied-weight restoration, checkpoint selection, and a preview that cannot spawn a worker. They use fixed test tensors and no optimizer updates. The broader `test_pipeline.py` suite includes actual training and was not run for this change.
