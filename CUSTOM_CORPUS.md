# Fine-tune with your own .txt corpus

The GUI accepts an ordinary **UTF-8 `.txt` file, up to 64 MiB**, and prepares it using a selected local checkpoint's exact tokenizer. Fine-tuning starts from that checkpoint's learned weights and inherits all its architecture settings, including context, vocabulary, RoPE and KV heads.

## GUI

1. Open **Data & training → 04b / Prepare your .txt corpus**.
2. Choose the inference checkpoint you want to fine-tune, such as `runs/brimv2c/best.pt`. **Use currently serving checkpoint** fills the current selection.
3. Select your `.txt` file, or enter/select a saved workspace-local `.txt` path.
4. Choose a fresh dataset directory and validation fraction, then click **Upload & prepare .txt**.
5. Watch **Job monitor**. Preparation creates the dataset without training the model.
6. In **05 / Fine-tune a checkpoint**, review the selected checkpoint/dataset, fresh run directory, passes, learning rate, microbatch and accumulation. Use **Preview fine-tuning budget**, then **Start fine-tuning** when ready.

Preparation fills the fine-tuning fields. After reopening the page, **Use this dataset for fine-tuning** restores them from the completed preparation job. Saved fine-tuning datasets are also listed in section 05; selecting one fills its original checkpoint path. Uploaded files remain listed for reuse.

**Stop job** cancels preparation, or asks training to save at an optimizer boundary. A stopped preparation can be run again with the same input/output paths; only a fully prepared dataset is published. **Resume fine-tuning run** restores the selected run's saved data, configuration, optimizer, schedule and weights. Choose the existing run directory before resuming. The resulting `best.pt` can be explicitly loaded into the playground or exported to GGUF.

## Text format and learning objective

Use your actual text, for example:

```text
The first passage discusses a subject in your preferred writing style.
It can span several lines.

This is another passage. Blank lines separate passages for the validation split.

Additional distinct passages provide more training and validation material.
```

- UTF-8 with or without a BOM, LF and CRLF are accepted. Empty/binary files are rejected.
- URLs are removed using the existing text policy. Unicode/whitespace normalization and literal special-token escaping follow the dialogue text preparation rules.
- Duplicate normalized passages and duplicate token sequences are removed before splitting.
- Multiple paragraphs are kept intact and assigned to deterministic, disjoint training/validation groups. The requested validation fraction applies to groups; the report records actual token counts/fraction.
- If only one distinct text block remains, it is divided into balanced, non-overlapping token chunks no longer than the checkpoint context. Duplicate chunks are removed. At least two distinct nonempty passages/chunks are required.
- Only the normalized text is tokenized, with the original BOS/EOS IDs. Original filename, raw-file SHA-256 and byte count remain provenance metadata. Uploads preserve the original bytes under unique paths in `data/custom-uploads/`.
- Short splits are supported with shuffled, variable-length batches. Training and validation stay separate even when a split is shorter than the model's context.

Custom text uses **next-token loss over all corpus text**. It does not infer which lines are user questions or assistant answers. The separate human-dialogue preparation flow continues to provide assistant-only supervised targets. A dialogue checkpoint retains its existing chat-format metadata after text fine-tuning; a base checkpoint remains a text-completion model.

Text training windows can cross paragraph boundaries within a split, with BOS/EOS marking each passage. Architecture and tokenizer are preserved, while fine-tuning updates model weights. File preparation does not update weights or fit a tokenizer. Pass budgets round up to whole optimizer updates; preview displays the resulting maximum token-equivalent passes, including rounding on very small corpora.

The dataset pins the selected checkpoint's SHA-256. If that checkpoint is replaced after preparation, use the original snapshot or prepare the corpus again before starting a new run. Resume instead uses the run's own `last.pt` and verifies dataset/tokenizer hashes, including metadata. Keep the prepared dataset available for resume.

## CLI

Prepare only:

```bash
.venv/bin/python -m scratchllm.custom_corpus \
  --checkpoint runs/brimv2c/best.pt \
  --input data/my-corpus.txt \
  --output data/my-custom-dataset \
  --validation-fraction 0.1
```

Start fine-tuning explicitly:

```bash
.venv/bin/python -m scratchllm.train \
  --init-from runs/brimv2c/best.pt \
  --data data/my-custom-dataset \
  --out runs/my-custom-model \
  --config configs/gtx1050ti-dialogue.json
```

The configuration supplies learning/budget settings; architecture is inherited from the checkpoint. GUI runs save their exact configuration and dataset paths automatically. For a CLI resume, supply the same data/config/run and replace `--init-from ...` with `--resume`.

## API and files

- `POST /corpus/upload?filename=my-corpus.txt`: raw file bytes as `application/octet-stream`; returns a unique workspace path, original filename, size and SHA-256. Uploading alone starts no job.
- `GET /corpus/uploads`: saved uploads and the upload size limit.
- `POST /jobs/custom-corpus`: supervised CPU preparation, using the body below.
- `GET /datasets`: `finetuning_datasets` includes custom text and dialogue datasets. Custom text datasets are excluded from the random-initialization base-training list.
- `POST /finetuning/plan`, `POST /jobs/finetune`, `POST /jobs/stop`, `GET /jobs`: shared budget, fine-tuning, cancellation and monitoring controls.

```json
{
  "input": "data/my-corpus.txt",
  "checkpoint": "runs/brimv2c/best.pt",
  "output": "data/my-custom-dataset",
  "validation_fraction": 0.1
}
```

Prepared datasets contain unchanged `tokenizer.json`, `train.bin` / `val.bin` uint32 token streams, split provenance JSONL, `report.json` and `meta.json` (`text-finetune-v1`). Only complete datasets are discoverable. Fine-tuned checkpoints retain source provenance and remain compatible with the existing serving and GGUF export controls.

## Verification

Focused tests cover uploads and UTF-8 boundaries, path/size validation, immutable source files, URL removal/deduplication, disjoint splits, short-stream evaluation, cancellation cleanup, supervised preparation, inherited model initialization, saved chat metadata and completed-run resume/hash validation. A real-browser test uploads a file, runs the CPU preparation worker, reopens the page and previews the budget; training start/stop/resume requests are intercepted. No optimizer steps are performed by these tests.

```bash
.venv/bin/python -m unittest discover -s tests -p test_custom_corpus.py -v
.venv/bin/python -m unittest discover -s tests -p test_custom_corpus_ui.py -v
```
