# Human dialogue fine-tuning with your existing vocabulary

Train the base model on Wikipedia and your other selected sites, then use its **`best.pt`** for the dialogue stage. Preparation copies its exact tokenizer; fine-tuning initializes from its learned weights. A board-only checkpoint does not acquire Wikipedia training simply by being selected here.

## GUI workflow

Refresh **http://127.0.0.1:8000/** and select **Data & training**.

- For a complete new run, choose your sites and settings, leave **After base training, prepare human dialogue and fine-tune** checked, and click **Start selected pipeline**. The server runs collect → prepare base → train base → prepare dialogue → fine-tune.
- For an existing base checkpoint, use **04 / Prepare human dialogue** with its `best.pt`, the dialogue datasets, and a fresh dataset directory. This downloads/prepares text without updating model weights.
- Use **05 / Fine-tune dialogue** to preview the inherited architecture and budget, then start explicitly in a fresh run directory. Defaults are **3 example passes, LR 0.00005**, microbatch 2 × accumulation 8, and early-stop patience 3.
- Stop and resume fine-tuning using **Stop job** and **Resume dialogue run**. Resume restores the saved dataset, optimizer, sampling state, and schedule; edited fields are ignored.
- Load the resulting dialogue `best.pt` explicitly. The playground's **Automatic from checkpoint** mode then uses chat formatting and history. **Clear** starts a new conversation.

Automatic paths retain both base and dialogue runs. The latter uses `data/dialogue-pipeline-<id>` and `runs/dialogue-pipeline-<id>`. A failure/cancellation prevents later stages from starting. Closing the GUI tab is safe; restarting the server interrupts the pipeline. An interrupted preparation may leave a partial directory; choose a fresh output when retrying. Completed downloads remain cached.

## Established human-written datasets

| Dataset | Selection | Attribution/license |
|---|---|---|
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English, positively reviewed, non-deleted, explicitly non-synthetic messages; highest-ranked eligible human reply per prompt; human-only ancestors | OpenAssistant contributors / LAION, Apache-2.0 |
| [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Human instruction/response pairs, including supplied reference context | Databricks and Wikipedia contributors, CC-BY-SA-3.0 |

These are dataset exports downloaded over HTTPS, independently of the Camoufox website crawler. Revisions, export sizes, and SHA-256 checksums are pinned in `scratchllm/dialogue.py`; successful downloads are cached in `data/dialogue-downloads`. The prepared directory includes source revisions, hashes, license/attribution, transformations, per-source counts, and readable examples. No remote model code or pretrained weights are used.

The pinned exports yielded 5,583 eligible OASST response examples and 15,009 Dolly examples before deduplication/context filtering. The usable count depends on your base tokenizer and context length.

## What is preserved and what learns

Preparation **copies the exact tokenizer JSON**: no BPE fitting, vocabulary expansion, or token-ID remapping. Its byte-level vocabulary also encodes words absent from the original corpus. Fine-tuning initializes all model weights from your selected base, with a fresh optimizer/schedule. Vocabulary preservation does not freeze the embedding weights or restrict all newly learned facts to the original sites.

`User:` and `Assistant:` markers are encoded with ordinary existing tokens. Each example contains conversation context and one target reply. Prompts, earlier replies, and padding use ignored loss targets; only the final assistant response and EOS contribute to the objective. Microbatch gradients and validation NLL are weighted by supervised target-token count. Sequences never cross conversations.

Validation groups whole OpenAssistant conversations and whole Dolly reference passages. Exact duplicate conversations and duplicate model-visible sequences are removed before splitting. Oldest complete history turns may be dropped to fit; a current question plus complete answer that exceeds context is skipped and reported. Answers are never cut off and falsely labeled complete.

Inference uses the same prompt-tokenization function as preparation. Chat drops oldest complete turns when the context fills and rejects a single question too long to fit. History lives in the browser/request, not shared server state. Raw text-completion mode remains selectable.

## CLI

Prepare dialogue without training a model:

```bash
python -m scratchllm.dialogue --checkpoint runs/response-v2/best.pt \
  --output data/prepared-dialogue-v2 --sources oasst1 dolly15k
```

Start training explicitly when ready:

```bash
python -m scratchllm.train --data data/prepared-dialogue-v2 \
  --config configs/dialogue.json --init-from runs/response-v2/best.pt --out runs/dialogue-v2
```

Architecture/context are inherited from the base. Prepared tokenizer, checkpoint fingerprint, and dimensions must agree. Dialogue budgets count **examples**, rather than packed-stream tokens:

`updates = ceil(passes × training_examples / (microbatch × accumulation))`

The final effective batch may slightly exceed the requested pass count. The default learning rate decays from 0.00005 to 0.000005. Warmup is shortened to fit the budget. Early stopping uses patience 3 and minimum improvement 0.001. `best.pt` retains the lowest fixed held-out **assistant-only NLL**, which is not directly comparable to full-stream base-model NLL.

Resume restores the original optimizer/RNG, sampling state, data/mask/boundary fingerprints, and stopping state. Do not also supply `--init-from`:

```bash
python -m scratchllm.train --data data/prepared-dialogue-v2 \
  --config configs/dialogue.json --out runs/dialogue-v2 --resume
```

Use the trained chat format from the command line:

```bash
python -m scratchllm.generate --checkpoint runs/dialogue-v2/best.pt --chat --prompt 'Explain how rain forms.'
```

## API

- `POST /jobs/dialogue`: `checkpoint`, fresh `output`, `sources`, optional `cache` and `validation_fraction`.
- `POST /finetuning/plan`: read-only checkpoint/data compatibility and budget preview.
- `POST /jobs/finetune`: `checkpoint`, prepared dialogue `data`, fresh `out`, and optimization settings. Resume with `out` and `resume: true`.
- `POST /chat`: `prompt`, optional alternating user/assistant `history` entries with `role`/`content`, plus normal generation settings. Requires a dialogue checkpoint.
- `POST /jobs/pipeline`: add `"dialogue": {"sources": ["oasst1", "dolly15k"]}` and optional `finetune` settings. The base checkpoint and all stage inputs are linked server-side. Omitting `dialogue` preserves the base-only API workflow.
- `GET /datasets`: dialogue datasets appear under `dialogue_datasets`, separately from base datasets, so they cannot become the default random-initialization dataset.

## Verification

```bash
python -m unittest discover -s tests -p test_dialogue.py -v
python -m unittest discover -s tests -p test_dialogue_ui.py -v
python -m unittest discover -s tests -p test_automated_pipeline.py -v
```

These checks do not fit a model or perform optimizer steps. They check frozen token IDs/weight initialization, human-source selection, conversation grouping, assistant-only targets and weighted evaluation, sampler/resume integrity, consistent chat formatting, browser controls, and five-stage cancellation with file-only workers. The default settings have not been quality-validated by training a new dialogue model.
# Custom conversation JSONL

In **04 / Prepare dialogue**, enter a workspace-local JSONL file containing
alternating `user` / `assistant` messages. The **Use Swinnish conversations
only** shortcut selects `data/swinnish-chat-2026-11-01/train.jsonl` and clears
the download selections. Choose a native inference `.pt` checkpoint and a
fresh dataset directory, prepare, then preview/start in section 05.

Each assistant turn becomes an example. Loss covers only its response and
EOS; user text and earlier history are masked. Optional `group` values keep
related conversations together; otherwise conversation content identifies
the group. Original synthetic/language/source metadata is retained. The chosen
file receives an internal grouped validation split; sibling files are not
automatically included. Downloads can be combined with the custom file by
keeping their checkboxes selected.

CLI: `python -m scratchllm.dialogue --checkpoint runs/swinny/best.pt --input data/swinnish-chat-2026-11-01/train.jsonl --output data/dialogue-swinnish`
