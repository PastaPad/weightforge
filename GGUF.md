# Export a checkpoint for llama.cpp

For **importing Hugging Face GGUF models and fine-tuning them on the site scrapers**, use the separate **Hugging Face / GGUF** tab and [HUGGINGFACE.md](HUGGINGFACE.md). The exporter described here handles native ScratchLLM `.pt` checkpoints.

The exporter writes a standalone **GGUF** containing the selected saved model's weights, architecture, original byte-level BPE vocabulary/merges, special-token IDs, and (for dialogue checkpoints) its chat template. It runs entirely on the CPU and uses a separate supervised worker, so a training or collection job can continue concurrently.

## GUI

In **Data & training → 06 / Export for llama.cpp**:

1. Choose a saved inference checkpoint from the list, enter its workspace-relative path, or use **Use serving checkpoint** / **Use checkpoint selected above**.
2. Choose **F16** (smaller file) or **F32** (full-precision weights).
3. Choose a fresh `.gguf` output path and click **Export GGUF**.
4. Watch the separate export progress/log. When complete, use **Download GGUF**, or open the output path directly in llama.cpp.

Select `best.pt`, not `last.pt`. The latter contains optimizer/RNG state for resuming training. Export reads the selected file rather than whichever model happens to be in GPU memory. Exporting a running job's `best.pt` captures its best saved validation checkpoint at that moment, not future updates.

**Stop export** affects only the export worker. A stopped export can be started again. The finished GGUF is published atomically after all tensors are written and the complete file is hashed. Existing outputs are never overwritten. Only one export runs at a time; its status survives a server restart. Export does not select a different serving model.

If the GUI says export is staged on an older running server, finish the active job before restarting `scratchllm.service`, then refresh. The CLI below can be used immediately, including while training is active.

## CLI

From this workspace, with the virtual environment activated:

```bash
python -m scratchllm.gguf_export \
  --checkpoint runs/brimv2c/best.pt \
  --output exports/brimv2c-f16.gguf \
  --precision f16
```

Choose any of your saved inference checkpoints. Use `--precision f32` and a different output filename for full-precision weights. `--progress-file runs/my-export.progress.json` optionally writes progress snapshots. Ctrl+C stops the export.

Run a current llama.cpp server with the resulting file, for example:

```bash
llama-server -m exports/brimv2c-f16.gguf --host 127.0.0.1 --port 8080 -c 512 -ngl 0
```

Here `-ngl 0` explicitly uses the CPU. Use a context size no greater than the checkpoint's trained context. A dialogue checkpoint supplies its chat template; a base checkpoint supplies a text-completion model. For dialogue, the template uses alternating user/assistant messages and the locally trained `User:` / `Assistant:` format with the original BOS/EOS IDs.

Further quantization is a separate llama.cpp operation, for example:

```bash
llama-quantize exports/brimv2c-f16.gguf exports/brimv2c-q8_0.gguf Q8_0
```

The Python exporter depends on `gguf>=0.19,<0.20`, installed through this project's dependencies. It does not require a llama.cpp executable to produce the file.

## Supported architectures and fidelity

| ScratchLLM architecture | GGUF architecture | Conversion |
|---|---|---|
| `modern` | `llama` | RMSNorm, grouped-query attention, adjacent-pair RoPE, SwiGLU, tied embeddings |
| Original `gpt` | `gpt2` | Learned positions, LayerNorm, packed QKV, GELU, tied embeddings; zero-filled attention biases where llama.cpp expects them |

- All dimensions come from the selected checkpoint; the exporter is not hard-coded to the 42M preset.
- The modern Q/K layout already matches GGUF Llama's adjacent-pair rotation. It is not subjected to the permutation used when converting Hugging Face split-half RoPE weights.
- The original GPT's exact GELU is evaluated by llama.cpp GPT-2 using its tanh approximation, so that architecture has a small activation-function difference even with F32 weights. This is recorded in export status and GGUF metadata.
- F16 stores matrices in half precision and keeps normalization/bias vectors in F32. F32 stores all exported tensors in F32. Neither option retrains the model.
- The vocabulary and merges remain the locally fitted tokenizer's, with their original IDs. The GGUF tokenizer family is `gpt2` / pre-tokenizer `gpt-2`, describing its byte-level BPE algorithm, not pretrained GPT-2 data or weights.
- Unsupported tokenizer transformations, incompatible shapes, untied embedding/output weights, non-finite values, and F16 overflow fail explicitly instead of publishing an incompatible file.
- Source checkpoint SHA-256, saved step, model configuration, tokenizer SHA-256, and exporter version are embedded. The completed file's SHA-256 appears in progress/API results.

The Python app's text normalization, URL removal, persistent SQLite memory, feedback, and optional review loop are application runtime behavior, not model weights. They do not become llama.cpp features merely by exporting the checkpoint. The chat template preserves the dialogue structure; equivalent application-side text cleanup must be applied separately if needed. The template also does not increase the model's trained context.

## API

- `GET /exports/checkpoints`: available local inference files under `runs/`, excluding `last.pt`.
- `POST /exports`: start a CPU export independently of the main job/GPU lock.
- `GET /exports`: export status, progress, recent log, and `export_ready`.
- `POST /exports/stop`: stop just the current export.
- `GET /exports/file?job_id=<id>`: download the completed file for the current export job.

Example `POST /exports` body:

```json
{
  "checkpoint": "runs/brimv2c/best.pt",
  "output": "exports/brimv2c-f16.gguf",
  "precision": "f16"
}
```

API paths must stay inside the workspace. The main training/collection status remains at `/jobs`; export state is stored separately at `runs/exports/latest.json`. Logs and progress files have unique IDs under `runs/jobs/`.

## Verification

Tested against **ggml-org/llama.cpp commit `f5027fb21307eb264aed483242603d68c87b0fe3`**, built CPU-only on 2026-10-23, with Python `gguf 0.19.0`.

The real llama.cpp tests load both architectures in F16/F32, compare token IDs for text/Unicode/whitespace/special tokens, apply the chat template, exercise chat completion, and compare next-token log-probabilities and greedy predictions to PyTorch. They use tiny random checkpoints and perform no model training.

A temporary snapshot of `runs/brimv2c/best.pt` (best saved step 3100, 6 × 256 modern decoder, vocabulary 8192) was also checked against actual llama.cpp CPU inference:

| Export | File size | Largest difference in tested top-50 log-probabilities |
|---|---:|---:|
| F32 | 39.77 MiB | 0.00000406 |
| F16 | 20.02 MiB | 0.00103 |

Tokenization, dialogue prompt IDs, and greedy predictions matched on the tested inputs. These are numerical conversion checks, not a response-quality evaluation. Temporary validation exports were removed; use the GUI or CLI to create your chosen deliverable.

```bash
python -m unittest discover -s tests -p test_gguf_export.py -v
python -m unittest discover -s tests -p test_gguf_ui.py -v
LLAMA_SERVER=/path/to/llama-server \
  python -m unittest discover -s tests -p test_gguf_llama_cpp.py -v
```

The core tests additionally cover atomic checkpoint replacement during export, no-overwrite publication, cancellation/failed-write cleanup, unsupported-tokenizer rejection, download scoping, persistence, and exports during an independent training reservation.
