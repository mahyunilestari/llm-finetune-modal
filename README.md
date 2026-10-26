# llm-finetune-modal

Fine-tuning and evaluation experiments for small open-weight language models
(7B–8B class), run as short detached jobs on [Modal](https://modal.com).

This is a personal research repo: a training loop, the Modal entrypoint that runs it,
and the config files I actually use. No serving, no public endpoints.

## What runs

| Piece | Detail |
|---|---|
| GPU | 1 x `GPU_RTX_PRO_6000` per job |
| Runtime | detached job, 24 h ceiling, one job at a time |
| Data | my own domain corpora (technical docs, support transcripts), loaded at start |
| Method | LoRA adapters on top of a frozen base model |
| Checkpointing | every N optimizer steps to a Modal Volume, resume from last checkpoint |
| Logging | step / loss / lr / GPU% / tokens-s / grad-norm / mem% to stdout |
| Eval | held-out split every 50 steps: validation loss + perplexity |

Nothing is persisted on Modal except checkpoints and adapters. Training data is pulled
from my own storage at job start and is deleted with the container.

## Why Modal

The workload is bursty — a few multi-hour GPU jobs per week, idle the rest of the time.
Detached jobs survive my laptop closing, Volumes give me durable checkpoints between
runs, and per-second billing means an idle week costs nothing. A reserved GPU instance
would be worse on every one of those axes.

## Layout

```
train.py                        training + eval loop (transformers + peft)
deploy_modal.py                 Modal app: image, GPU, timeout, Volume mount
configs/rtxpro6000-1x_finetune.yaml   the config used for RTX PRO 6000 runs
logs/sample-run.log             a trimmed run log (format = what the job prints)
```

## Run it

```bash
pip install -r requirements.txt
modal run --detach deploy_modal.py            # detached: survives client exit
modal app logs <app-id>                       # follow progress
```

Resume after a preemption or a client disconnect:

```bash
modal run --detach deploy_modal.py --resume /output/checkpoint-3270
```

The app is intentionally built so a killed client does **not** kill the job. Re-running
the same config while a job is live would queue a second one, so I check
`modal app list` first.

## Log format

```
[TRAIN] Step 003277 | Loss: 0.1979 | LR: 2.39e-04 | GPU: 95% | Tokens/s: 3161 | GradNorm: 0.719 | Mem: 80%
[INFO] Checkpoint saved: /output/checkpoint-3270.pt (5.5 GB)
[EVAL] Validation loss: 0.3012 | Perplexity: 1.35
```

See `logs/sample-run.log` for a real excerpt.

## Notes

- Base model weights are pulled at image build time from the Hugging Face Hub.
- Adapters and checkpoints land on a Modal Volume named `finetune-output`.
- Dataset paths are configured per run; the datasets themselves are not in this repo.
