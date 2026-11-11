# Hugging Face GGUF import and scraper fine-tuning

Open **http://127.0.0.1:8000/** and select **Hugging Face / GGUF**. This workflow imports model weights you choose, uses their original tokenizer, and trains local **PEFT LoRA adapters** with PyTorch. The original from-scratch workflow remains in **Data & training**.

The server also starts when its configured native checkpoint is missing. Collection, preparation and Hugging Face tools remain available; the playground displays **No native checkpoint loaded** until you explicitly load an existing `.pt` file.

## GUI workflow

1. **Import a GGUF model:** enter a Hugging Face model repository such as `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, and click **List GGUF files**. Choose a complete language-model file or shard set, a fresh output directory under `models/`, and F16/F32 storage. Click **Download & import GGUF**. The repository revision is pinned, files are checked against their Hub SHA-256, and conversion runs on CPU. When complete, click **Use this result in the next stage**.
2. **Collect scraper text (optional):** select `jewish.win`, `soyjak.st`, `swedishwin.com`, `soyjakwiki.org`, and/or offline Wikipedia. These controls use the existing collectors, shared browser/ZIM settings, and saved crawl state. **Collection / ZIM settings** opens the full settings. If you already have scraped JSONL text, go directly to section 03. Wikipedia still reads only a local ZIM; browser/curl_cffi behavior and the default robots override are unchanged.
3. **Prepare scraper text:** choose **Source JSONL or .txt file** directly in section 03. The list includes raw corpora and existing prepared `train.jsonl`/`val.jsonl` files; workspace-relative and full paths inside the workspace are accepted. UTF-8 `.txt` paths also work: upload through **Data & training → 02 / Prepare dataset** and copy the saved path here. For short custom sentences, set **Minimum document / article chars** to **1**. **Use collection output above** copies the collection path when wanted. Select the imported model, a fresh dataset directory, context, and validation fraction. Preparation removes URLs and duplicate text, splits whole source groups, preserves provenance, and encodes only the cleaned text with the imported tokenizer. Click **Use this result in the next stage** after preparation.
4. **Fine-tune:** select the imported base and its prepared dataset, a fresh run directory, passes, learning rate, batch/accumulation, and LoRA rank/alpha. **Preview budget & memory** starts no job. Click **Start imported-model fine-tuning** when ready. The shared monitor shows progress and validation NLL. **Stop current job** requests a resumable save; **Resume imported-model run** restores the run directory's original settings.
5. **Merge the result:** select the completed/stopped run to download its best adapter, tokenizer and provenance. **Merge best adapter into full model** produces a complete Hugging Face directory with `.safetensors` weights, configuration and tokenizer/chat template. Click **Use this result in the next stage** to select the merged directory for export.
6. **Export merged model to GGUF:** select that complete model directory, a fresh `.gguf` output path and F16, F32 or Q8_0 storage. Click **Export complete model to GGUF**. CPU conversion uses llama.cpp and checks the output's architecture, tokenizer IDs, special-token policy, chat template and floating tensors before atomic publication. Use **Download exported GGUF** in the shared monitor when finished. This also works with an already merged model; repeat training or merging is unnecessary.

Imports, collection, preparation, fine-tuning, merging and HF GGUF export use the supervised main worker, one job at a time. Closing the tab does not stop the worker. Generation is suspended during fine-tuning to free GPU memory. Stop also terminates an active GGUF converter and removes unpublished output. The selected native playground checkpoint stays explicit; imported HF directories and adapters are used through Transformers/llama.cpp rather than the native `.pt` loader.

Use **Resume this GGUF import** after a download/conversion interruption. The same settings/output restore its pinned revision and reuse cached download fragments. Preparation/merging can be restarted against an unpublished output. Only complete models/datasets are published. Existing nonempty outputs are rejected.

## Installation and compatibility

```bash
.venv/bin/python -m pip install -e '.[huggingface]'
```

The tested extra uses Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2, Hugging Face Hub 2.2.0, filelock 3.32.7, and safetensors 0.8.0. The readiness check includes `filelock`, which the Hub downloader requires. Downloads use the standard Hub cache under `data/huggingface-cache`; Hub login credentials or `HF_TOKEN` in the server environment support repositories requiring authentication. No repository Python code is executed (`trust_remote_code=False`).

Supported GGUF architecture identifiers: `llama`, `mistral`, `qwen2`, `qwen3`, `phi3`, `gpt2`, `gemma2`, `gemma3`, `falcon`, `bloom`, `stablelm`, `starcoder2`, `nemotron`. These are text causal-model imports; unsupported architectures, missing/incompatible tensors, unsupported tokenizers and out-of-vocabulary IDs fail explicitly. Some model variants within a family may require upstream loader support. Every converted vocabulary ID is checked; GGUF special tokens, BOS policy, EOS and chat template are retained. Shard sets are joined without changing packed tensor blocks before conversion.

GGUF is an inference storage format. Import fully dequantizes its weights into F16/F32 safetensors. This cannot recover precision already lost to quantization. Transformers' GGUF loader is explicitly put into full-dequantization mode; the published model contains ordinary trainable tensors.

### GTX 1050 Ti / 4 GB

- Start with approximately **0.5B parameters**, context **256**, microbatch **1**, accumulation **8**, rank **8**, alpha **16**, learning rate **0.0001** and **3 passes**. Fit depends on architecture/vocabulary, not just parameter count.
- **QLoRA / CUDA:** reloads the reconstructed base into bitsandbytes NF4 with double quantization, freezes it, and trains FP32 adapters. Computation stays FP32 for Pascal. CUDA compute capability 6.0+ is required; this machine is 6.1.
- **LoRA / CUDA or CPU:** freezes the FP32 base and trains FP32 adapters. This uses more model memory. Both modes enable activation checkpointing and `AdamW(foreach=False)`.
- CPU conversion estimates roughly `8 × stored parameter count + 512 MiB` available RAM, plus `2 × count` (F16) or `4 × count` (F32) output disk and metadata allowance. The downloaded GGUF and reconstructed model both consume disk; split GGUFs temporarily need joined-file space too.
- Fine-tuning previews include frozen weights, unquantized embedding/output layers, adapter/optimizer state, an activation allowance and overhead. Worker preflight rejects estimates beyond the usable budget. Actual memory varies; the preview is an estimate rather than a guarantee. Smaller files from heavier GGUF quantization do not make full CPU reconstruction equally small.

## Dataset and training behavior

- Existing prepared text such as `data/prepared-pipeline-<id>/train.jsonl` is reusable. Older files omitted per-row extraction markers, so their adjacent `meta.json`, `hf-meta.json`, or `report.json` must identify `article-post-text-v4` grouped preparation. Current prepared output retains extraction/text-policy markers in each record. Explicitly obsolete crawler versions are rejected. URL cleanup is reapplied even when the old dataset predates it.
- Selecting `train.jsonl` uses that selected file as a corpus and creates a fresh grouped train/validation split within it. Its sibling `val.jsonl` is not automatically included. The original text and token files are preserved; token IDs are newly encoded for the imported model.
- Source extraction remains **`article-post-text-v4`**, with **`user-text-no-links-v1`** cleanup: user-written articles/posts, no site UI/media, URLs stripped from training text, and source identity retained separately.
- Prepared `train.bin`/`val.bin` contain little-endian uint32 IDs. `hf-meta.json` records tokenizer/model identity, corpus SHA-256, split counts, token hashes, context, objective and per-source report. These datasets are excluded from native random-initialization training lists.
- Each document ends with its original EOS and receives BOS when specified by the GGUF tokenizer. Nonoverlapping token windows may cross document boundaries; EOS is the separator. Partial windows retain valid targets and ignore padding.
- Objective: next-token prediction on **all cleaned scraper text**. This adapts the model to the corpus distribution; it does not infer assistant-response labels from arbitrary posts. Existing chat templates are preserved for later inference.
- Updates are `ceil(passes × window_count / (microbatch × accumulation))`. Whole updates can round a small dataset's exposure upward. Accumulation is target-token weighted, validation uses fixed windows with dropout off, and the best held-out NLL selects the adapter. Default early stopping is three checks without a meaningful improvement of 0.001.
- A fine-tuning run stores `hf-run.json`, `metrics.jsonl`, atomic `best.json`/`last.json` pointers, and complete hashed adapter/optimizer/RNG/sampler snapshots under `checkpoints/`. Model/tokenizer files and dataset hashes are verified before training/resume. Resume restores its saved configuration, not newly edited form fields. Finished/early-stopped runs keep their stopping decision.
- Stop finishes the current optimizer update, evaluates and saves. If a worker cannot stop within the supervisor's grace period, it is terminated; resume then uses its last complete snapshot. Checkpoints are written at validation intervals as well as on stop.

## API and CLI

| Endpoint | Purpose |
|---|---|
| `GET /huggingface/status` | Dependency versions, architecture support, RAM/disk/GPU |
| `POST /huggingface/repository` | List GGUF choices for `{repo, revision}` |
| `GET /huggingface/assets` | Imported models, scraper corpora, HF datasets and runs |
| `POST /jobs/hf-import` | Download, verify and reconstruct the chosen GGUF |
| `POST /jobs/hf-prepare` | Prepare JSONL or UTF-8 .txt with the imported tokenizer |
| `POST /huggingface/plan` | Fine-tuning budget and memory estimate |
| `POST /jobs/hf-finetune` | Explicit start, or `{out, resume: true}` |
| `GET /jobs`, `POST /jobs/stop` | Shared status, logs and cancellation |
| `GET /huggingface/adapter?run=runs/my-adapter` | Download best PEFT adapter archive |
| `POST /jobs/hf-merge` | Merge best adapter into a complete HF directory |
| `POST /jobs/hf-export` | Export `{model, output, precision}` to a standalone GGUF |
| `GET /huggingface/gguf?job_id=...` | Download the current successful HF GGUF export |

Example request files for the CLI (all paths relative to this workspace):

```json
{
  "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "revision": "main",
  "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "output": "models/huggingface/qwen-small",
  "precision": "f16"
}
```

```bash
python -m scratchllm.huggingface --stage import --request import.json
```

```json
{
  "model": "models/huggingface/qwen-small",
  "input": "data/raw-text-v4.jsonl",
  "output": "data/qwen-sites",
  "context": 256,
  "validation_fraction": 0.1
}
```

```bash
python -m scratchllm.huggingface --stage prepare --request prepare-hf.json
```

```json
{
  "model": "models/huggingface/qwen-small",
  "data": "data/qwen-sites",
  "out": "runs/qwen-sites",
  "device": "cuda",
  "mode": "qlora",
  "rank": 8,
  "alpha": 16,
  "batch_size": 1,
  "accumulation": 8,
  "max_epochs": 3
}
```

```bash
# This explicitly starts optimization; run when ready.
python -m scratchllm.hf_train --request train-hf.json
```

Set `"resume": true` and the original `out` in a request to resume. CPU use requires `"device": "cpu", "mode": "lora"`. CLI operations optionally accept `--progress-file path.json`.

## Convert a merged model back to GGUF

Section **05** saves HF `.safetensors`, not a native `.pt` checkpoint or GGUF. Section **06** creates the standalone GGUF file. F16 halves the weight size relative to F32; Q8_0 adds 8-bit quantization. The GUI and CLI preserve existing files and leave the source directory intact. Source weights are checked against the merge/import manifest before and after export.

The converter is installed in `tools/llama.cpp` on this machine. To reproduce its pinned setup:

```bash
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
git -C tools/llama.cpp sparse-checkout init --cone
git -C tools/llama.cpp sparse-checkout set conversion gguf-py requirements
git -C tools/llama.cpp checkout b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9
.venv/bin/python -m pip install -e '.[huggingface]'
```

`SCRATCHLLM_LLAMA_CPP` can select another local converter checkout. The subprocess uses that checkout's matching `gguf-py`, CPU-only execution and offline model/tokenizer loading. Its known pre-tokenizer and architecture mappings determine supported export formats; unsupported mappings fail with details in the job log. `sentencepiece` is included in the HF extra because the converter checks it even for some BPE models.

Merge in the GUI first, or use a request such as:

```json
{"run": "runs/qwen-sites", "output": "models/huggingface/qwen-sites-merged", "precision": "f16"}
```

```bash
python -m scratchllm.huggingface --stage merge --request merge-hf.json
python -m scratchllm.hf_export --model models/huggingface/qwen-sites-merged \
  --output exports/qwen-sites-f16.gguf --precision f16
llama-server -m exports/qwen-sites-f16.gguf --host 127.0.0.1 --port 8080 -c 256
```

Use a llama.cpp runtime supporting the imported architecture. A PEFT adapter zip alone is not a standalone GGUF. QLoRA merging combines the adapter with the reconstructed base; NF4 training introduces additional approximation. Python-side persistent memory and runtime URL cleanup remain application behavior.

## Verification without model training

Focused tests use tiny random fixtures: GGUF F16/F32 reconstruction, tokenizer/chat/logit equivalence, Q8 packed-shard joining, frozen scraper tokenization and group separation, checksum/revision pinning, adapter-only gradient connectivity, zero-update cancellation snapshots/resume, merge/download, supervised preparation, and browser controls. Optimizer steps are forbidden in the tests; browser training launches are intercepted.

The GTX 1050 Ti also passed a real NF4 QLoRA forward/backward connectivity check: adapter gradients were finite/nonzero, base weights remained frozen, and no weights were updated. This verifies the CUDA path, not the fit or quality of larger models.

A live Hub download/import was verified with `ggml-org/models`, revision `499bc8821c6b12b4e53c5bffcb21ec206f212d81`, file `tinyllamas/stories260K.gguf` (1,185,376 bytes). Its 292,800-parameter Llama model, 512-token SentencePiece vocabulary and BOS policy reloaded successfully and produced finite inference logits. The temporary download and reconstructed model were removed after verification. The repository browser was also checked against `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.

```bash
python -m unittest discover -s tests -p test_huggingface.py -v
python -m unittest discover -s tests -p test_huggingface_ui.py -v
python -m unittest discover -s tests -p test_hf_export.py -v
```
