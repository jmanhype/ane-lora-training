#!/usr/bin/env python3
"""
End-to-end LoRA fine-tuning with ANE gradient dispatch.

Proves the full stack: MLX forward → ANE fused gradient kernel → weight update.
Runs on a small dataset with loss tracking to show actual learning.

Usage:
    cd /path/to/ane-lora-training
    python examples/train_e2e.py

    # Custom model/steps
    ANE_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \
    ANE_STEPS=20 ANE_LORA_RANK=8 ANE_LORA_LAYERS=2 \
    python examples/train_e2e.py
"""
import os
import sys
import time

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

# ---------- Config ----------
MODEL_NAME = os.environ.get("ANE_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
LORA_RANK = int(os.environ.get("ANE_LORA_RANK", "8"))
LORA_LAYERS = int(os.environ.get("ANE_LORA_LAYERS", "2"))
STEPS = int(os.environ.get("ANE_STEPS", "20"))
LR = float(os.environ.get("ANE_LR", "1e-4"))
BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "bridge", "libane_bridge.dylib"))

# ---------- Training data ----------
# Simple conversational pairs for demonstration
TRAIN_DATA = [
    "The Apple Neural Engine is a dedicated machine learning accelerator.",
    "LoRA stands for Low-Rank Adaptation, a parameter-efficient fine-tuning method.",
    "The ANE can process matrix operations at very low power consumption.",
    "Fine-tuning adapts a pre-trained model to a specific task or domain.",
    "Apple Silicon integrates CPU, GPU, and Neural Engine on one chip.",
    "Gradient computation can be offloaded from GPU to ANE for power savings.",
    "MLX is Apple's machine learning framework optimized for Apple Silicon.",
    "The packed IOSurface pattern enables compile-once ANE kernel dispatch.",
    "LoRA adds low-rank matrices A and B to frozen pre-trained weights.",
    "The fused gradient kernel computes all LoRA gradients in one ANE dispatch.",
]

# ---------- Load model ----------
print(f"Loading {MODEL_NAME}...")
t0 = time.time()
model, tokenizer = load(MODEL_NAME)
print(f"Loaded in {time.time()-t0:.1f}s")

# ---------- Apply LoRA ----------
layers = model.model.layers
total_layers = len(layers)
start = max(0, total_layers - LORA_LAYERS)

for i in range(start, total_layers):
    attn = layers[i].self_attn
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        orig = getattr(attn, proj, None)
        if orig and not isinstance(orig, LoRALinear):
            setattr(attn, proj, LoRALinear.from_base(orig, r=LORA_RANK))

lora_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters()))
total_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"LoRA: {lora_params:,} trainable / {total_params:,} total ({100*lora_params/total_params:.2f}%)")

# ---------- ANE integration ----------
ane_active = False
try:
    from ane_lora_kernels import (
        ANELoRAKernels, set_ane_kernels, replace_lora_with_ane, get_dispatch_stats)

    kernels = ANELoRAKernels(BRIDGE_PATH)
    err = kernels.verify_conv()
    if err < 0.1:
        set_ane_kernels(kernels)
        replaced = replace_lora_with_ane(model)
        ane_active = replaced > 0
        print(f"ANE: {replaced} layers replaced, fused gradient dispatch active")
    else:
        print(f"ANE: verify failed (err={err:.4f}), using MLX GPU")
except Exception as e:
    print(f"ANE: unavailable ({e}), using MLX GPU")

# ---------- Loss function ----------
def loss_fn(model, tokens):
    logits = model(tokens[None, :-1])
    targets = tokens[1:]
    return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

loss_and_grad = nn.value_and_grad(model, loss_fn)

# ---------- Training loop ----------
print(f"\n{'='*60}")
print(f"  E2E LoRA Training")
print(f"  Model:  {MODEL_NAME}")
print(f"  LoRA:   rank={LORA_RANK}, layers={LORA_LAYERS}")
print(f"  ANE:    {'ACTIVE (fused kernel)' if ane_active else 'OFF (MLX GPU)'}")
print(f"  Steps:  {STEPS}, LR={LR}")
print(f"{'='*60}\n")

losses = []
step_times = []
ane_dispatches_total = 0

for step in range(STEPS):
    # Use same sentence repeatedly to show loss convergence
    text = TRAIN_DATA[0] if step < STEPS // 2 else TRAIN_DATA[1]
    tokens = mx.array(tokenizer.encode(text))

    if len(tokens) < 2:
        continue

    t0 = time.time()

    pre_dispatches = 0
    if ane_active:
        pre_dispatches = get_dispatch_stats().get("ane_dispatches", 0)

    loss, grads = loss_and_grad(model, tokens)
    mx.eval(loss, grads)

    # SGD update
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    grad_flat = dict(mlx.utils.tree_flatten(grads))
    updates = [(k, trainable[k] - LR * grad_flat[k]) for k in trainable if k in grad_flat]
    if updates:
        model.load_weights(updates, strict=False)
        mx.eval(model.parameters())

    dt = time.time() - t0
    loss_val = float(loss)
    losses.append(loss_val)
    step_times.append(dt)

    ane_disp = 0
    if ane_active:
        post = get_dispatch_stats().get("ane_dispatches", 0)
        ane_disp = post - pre_dispatches
        ane_dispatches_total += ane_disp

    engine = f"ANE×{ane_disp}" if ane_disp > 0 else "GPU"
    print(f"  step {step+1:3d}/{STEPS}  loss={loss_val:.4f}  "
          f"time={dt*1000:.0f}ms  tokens={len(tokens)}  [{engine}]")

# ---------- Summary ----------
print(f"\n{'='*60}")
print(f"  Training Complete")
print(f"{'='*60}")
print(f"  Steps:       {STEPS}")
print(f"  Loss:        {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[-1]-losses[0]:+.4f})")
print(f"  Avg time:    {np.mean(step_times)*1000:.0f} ms/step")
if ane_active:
    stats = get_dispatch_stats()
    print(f"  ANE total:   {stats.get('ane_dispatches', 0)} dispatches")
    print(f"  ANE fallback:{stats.get('fallback_dispatches', 0)} dispatches")
print(f"  Learning:    {'YES ✓' if losses[-1] < losses[0] else 'NO ✗'}")
print()

# ---------- Save loss curve ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Save raw data as JSON for reproducibility
data_path = os.path.join(REPO_ROOT, "loss_data.json")
with open(data_path, "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_layers": LORA_LAYERS,
        "lr": LR,
        "steps": STEPS,
        "ane_active": ane_active,
        "losses": losses,
        "step_times": step_times,
        "avg_ms": float(np.mean(step_times) * 1000),
    }, f, indent=2)
print(f"  Saved loss data to {data_path}")

# Plot loss curve
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(range(1, len(losses) + 1), losses, color="#2563eb", linewidth=2)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"ANE LoRA Training — Loss Curve\n{MODEL_NAME}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f"{losses[0]:.2f}", (1, losses[0]), fontsize=10,
                 xytext=(10, 10), textcoords="offset points")
    ax1.annotate(f"{losses[-1]:.2f}", (len(losses), losses[-1]), fontsize=10,
                 xytext=(-40, 10), textcoords="offset points")

    # Step time
    ax2.plot(range(1, len(step_times) + 1),
             [t * 1000 for t in step_times], color="#16a34a", linewidth=1.5, alpha=0.7)
    ax2.axhline(y=float(np.mean(step_times) * 1000), color="#dc2626",
                linestyle="--", linewidth=1.5, label=f"avg={np.mean(step_times)*1000:.0f}ms")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title("Step Time", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    engine_label = "ANE fused kernel" if ane_active else "MLX GPU"
    fig.suptitle(f"LoRA rank={LORA_RANK}, layers={LORA_LAYERS}, LR={LR} — [{engine_label}]",
                 fontsize=11, y=0.02, color="gray")
    plt.tight_layout()

    plot_path = os.path.join(REPO_ROOT, "loss_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {plot_path}")
except ImportError:
    print("  (matplotlib not installed — skipping plot)")
