# Executed run

## Corpus collection

Camoufox was installed and run with 12 navigation attempts per site, depth 2, and a minimum three-second navigation interval.

| Source | Observed result |
|---|---|
| `jewish.win` | 12 navigation attempts; 11 unique documents saved |
| `soyjak.party` | Homepage returned HTTP 404; no documents collected |
| `swedishwin.com` | Homepage disallowed by robots.txt; no documents collected |
| `soyjakwiki.org` | robots.txt returned HTTP 403; origin skipped |

Raw documents and provenance are in `data/raw.jsonl`. **This checkpoint learned only from the successfully collected `jewish.win` text, not from all four sources.** Collection from the other sites remains blocked by the results above.

Preparation used a requested vocabulary of 2,048 and a validation fraction of 0.2. The actual learned vocabulary contains **1,251 tokens**. Document splitting produced 7 training documents (7,279 tokens) and 4 validation documents (9,783 tokens). Hash-based splitting is by document, so token counts need not match the requested fraction, especially in such a small corpus.

## GPU training

Command executed:

```bash
.venv/bin/python -m scratchllm.train --config configs/demo.json --out runs/demo-fp32
```

- GPU: NVIDIA GeForce GTX 1050 Ti, 4 GB, compute capability 6.1.
- PyTorch: `2.14.0+cu126`; precision: FP32.
- Randomly initialized parameters: **5,085,952**.
- Architecture: 6 blocks, width 256, 8 heads, context 128.
- Training: 40 updates, 512 sampled tokens per update, 20,480 sampled tokens total.
- Best sampled validation loss: **5.1903** at step 30; perplexity **179.52**.
- Final sampled validation loss: **5.3447** at step 40; perplexity **209.50**.
- Checkpoints: `runs/demo-fp32/best.pt` and `runs/demo-fp32/last.pt`.
- Metrics: `runs/demo-fp32/metrics.jsonl`.

An initial diagnostic run in `runs/demo` used emulated BF16 because PyTorch reported BF16 support despite the GPU lacking native BF16. Automatic precision selection was corrected to check native support, and the complete demonstration was rerun in FP32 as reported above.

This is an end-to-end demonstration, not a well-trained assistant. Greedy generation from `The history of` produced repeated `[Quote]` markers from the source pages. More clean, diverse text and substantially longer training are needed for useful language generation. Validation samples are small and differ between evaluations, so these values are noisy and not a strong quality estimate.

## Use the trained checkpoint

```bash
source .venv/bin/activate
python -m scratchllm.generate --checkpoint runs/demo-fp32/best.pt --prompt "The history of"
python -m scratchllm.serve --checkpoint runs/demo-fp32/best.pt --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/complete \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The history of","max_new_tokens":32,"temperature":0.8}'
```

The API was verified with the actual checkpoint on CUDA. The offline test suite also passed all three tests, covering causal isolation and learning, extraction/scope, and the preparation → training → resume loading → serving pipeline.

The current demo run has finished all configured steps; `--resume` simply loads that completed run. To perform a longer fresh training run, use `configs/tiny.json` and `runs/tiny` as described in [README.md](README.md). Collect and clean more data before investing in a long run.
