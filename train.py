#!/usr/bin/env python3
"""train.py — LoRA fine-tuning + eval loop for small open-weight models.

Prints one line per optimizer step so a detached run is readable from
`modal app logs` without a dashboard:

    [TRAIN] Step 000123 | Loss: 0.4123 | LR: 2.10e-04 | GPU: 94% | Tokens/s: 2450 | GradNorm: 0.830 | Mem: 88%
    [INFO] Checkpoint saved: /output/checkpoint-150.pt (5.8 GB)
    [EVAL] Validation loss: 0.3211 | Perplexity: 1.38

Usage (inside the Modal container):
    python train.py --config configs/rtxpro6000-1x_finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")


def gpu_util() -> int:
    """Rough GPU utilisation percentage without adding a pynvml dependency."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0


def gpu_mem() -> int:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return int(round(100 * used / total))
    except Exception:
        return 0


class StepLogger(TrainerCallback):
    """One compact line per step + checkpoint/eval lines, matching the README format."""

    def __init__(self, ckpt_every: int, tokens_seen: list[int]):
        self.ckpt_every = ckpt_every
        self.tokens_seen = tokens_seen
        self.t0 = time.time()
        self.steps = 0

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self.steps += 1
        loss = state.log_history[-1].get("loss") if state.log_history else None
        if loss is None:
            return
        elapsed = max(time.time() - self.t0, 1e-6)
        tps = int(self.tokens_seen[0] / elapsed)
        lr = state.log_history[-1].get("learning_rate", args.learning_rate)
        gn = state.log_history[-1].get("grad_norm", 0.0) or 0.0
        print(
            f"[TRAIN] Step {state.global_step:06d} | Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"GPU: {gpu_util()}% | Tokens/s: {tps} | GradNorm: {gn:.3f} | Mem: {gpu_mem()}%",
            flush=True,
        )

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        step = state.global_step
        path = os.path.join(OUTPUT_DIR, f"checkpoint-{step}.pt")
        size_gb = 0.0
        if os.path.isdir(os.path.join(OUTPUT_DIR, f"checkpoint-{step}")):
            for root, _, files in os.walk(os.path.join(OUTPUT_DIR, f"checkpoint-{step}")):
                size_gb += sum(os.path.getsize(os.path.join(root, f)) for f in files) / 1e9
        print(f"[INFO] Checkpoint saved: {path} ({size_gb:.1f} GB)", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        metrics = metrics or {}
        val = metrics.get("eval_loss")
        if val is None:
            return
        print(
            f"[EVAL] Validation loss: {val:.4f} | Perplexity: {math.exp(val):.2f}",
            flush=True,
        )


def build_dataset(tok, cfg):
    ds = load_dataset(
        "json",
        data_files={"train": cfg["data"]["train_file"], "validation": cfg["data"]["val_file"]},
    )

    def tok_fn(batch):
        out = tok(
            batch["text"],
            truncation=True,
            max_length=cfg["data"]["max_seq_len"],
            padding=False,
        )
        return out

    ds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="path to a checkpoint dir to resume from")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] config {json.dumps(cfg)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(tok, cfg)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    tokens_seen = [0]

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type=cfg["train"]["scheduler"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        max_steps=cfg["train"]["max_steps"],
        logging_steps=1,
        save_steps=cfg["train"]["save_every"],
        eval_steps=cfg["train"]["eval_every"],
        evaluation_strategy="steps",
        bf16=True,
        report_to=[],
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        callbacks=[StepLogger(cfg["train"]["save_every"], tokens_seen)],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()
