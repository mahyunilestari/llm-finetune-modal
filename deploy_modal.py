"""deploy_modal.py — Modal entrypoint for a single detached fine-tuning run.

    modal run --detach deploy_modal.py
    modal run --detach deploy_modal.py --resume /output/checkpoint-3270

One GPU, one job, 24 h ceiling. Checkpoints go to a persistent Volume so a
preemption costs at most `save_every` steps of work.
"""
from __future__ import annotations

import os
import subprocess
import sys

import modal

APP_NAME = "llm-finetune"
VOLUME_NAME = "finetune-output"
GPU = "RTX-PRO-6000:1"
TIMEOUT = 24 * 60 * 60

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers==4.44.2",
        "peft==0.12.0",
        "datasets==2.21.0",
        "accelerate==0.33.0",
        "pyyaml==6.0.2",
    )
    .add_local_dir(".", remote_path="/workspace", copy=True)
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=2.0,
    memory=32768,
    timeout=TIMEOUT,
    volumes={"/output": vol},
)
def run_training(config: str = "configs/rtxpro6000-1x_finetune.yaml", resume: str | None = None):
    os.chdir("/workspace")
    cmd = [sys.executable, "train.py", "--config", config]
    if resume:
        cmd += ["--resume", resume]
    print(f"[INFO] launching: {' '.join(cmd)}", flush=True)
    # inherit stdout so step lines land in `modal app logs`
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(config: str = "configs/rtxpro6000-1x_finetune.yaml", resume: str = ""):
    print(f"[INFO] deploying detached run (1x RTX PRO 6000) config={config}")
    run_training.remote(config=config, resume=resume or None)
