# ANE LoRA Training: Fused Neural Engine Gradient Kernels for Apple Silicon

**The first public hybrid MLX + ANE system with fused LoRA gradient dispatch and end-to-end training.**

LoRA gradient computation — all 4 matrix multiplications per adapter module — fused into a single ANE dispatch via packed-IOSurface dynamic matmul. Compile once, update weights via IOSurface each step. Zero recompile, zero handle leak.

```
  E2E LoRA Training
  Model:  mlx-community/Qwen2.5-3B-Instruct-4bit
  LoRA:   rank=8, layers=2
  ANE:    ACTIVE (fused kernel)

  step   1/20  loss=4.0742  time=254ms  tokens=11  [ANE×8]
  step   2/20  loss=4.0742  time=159ms  tokens=11  [ANE×8]
  ...
  step  10/20  loss=4.0625  time=149ms  tokens=11  [ANE×8]
  ...
  step  20/20  loss=2.8457  time=161ms  tokens=18  [ANE×8]

  Loss:        4.0742 → 2.8457 (Δ=-1.2285)
  Avg time:    163 ms/step
  ANE total:   160 dispatches
  ANE fallback:0 dispatches
  Learning:    YES ✓
```

## What Makes This Different

| Feature | This Repo | maderix/ANE | Espresso | CoreML |
|---------|-----------|-------------|----------|--------|
| ANE gradient computation | **Yes** | No (inference only) | No public gradients | No training |
| Fused 4-matmul kernel | **Yes** (1 dispatch/module) | N/A | Multi-layer fusion | N/A |
| Compile-once caching | **Yes** (zero recompile) | N/A | Unknown | N/A |
| MLX VJP integration | **Yes** (transparent) | No | No | No |
| Python API | **Yes** (ctypes) | ObjC only | ObjC only | Swift |
| End-to-end training | **Yes** (proven) | No | No public demo | No |
| Live fine-tuning daemon | **Yes** (HTTP API) | No | No | No |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLX Inference (GPU)                       │
│                                                             │
│  Forward:  base(x) + scale * (x @ lora_a @ lora_b)         │
│  Loss:     cross_entropy(pred, target)                      │
│  Backward: nn.value_and_grad → mx.custom_function VJP       │
│            dx → stays on GPU (chain rule)                   │
│            d_lora_a, d_lora_b → dispatched to ANE ──────────│──►
│                                                             │
└─────────────────────────────────────────────────────────────┘
                                                         │
                              ┌───────────────────────────▼──┐
                              │     ANE Fused Kernel (~2.8W) │
                              │                              │
                              │  Single IOSurface packing:   │
                              │    dy^T + x^T + B^T + A      │
                              │                              │
                              │  4 matmuls in 1 MIL program: │
                              │    tmp  = B @ dy^T           │
                              │    axT  = A^T @ x^T          │
                              │    d_A  = x^T @ tmp^T        │
                              │    d_B  = axT @ dy           │
                              │                              │
                              │  Compile once, cached forever │
                              │  Returns: d_A, d_B (fp32)    │
                              └──────────────────────────────┘
```

## Key Technical Innovations

### 1. Packed-IOSurface Dynamic Matmul

Standard ANE matmul fails at eval. Conv works but requires BLOBFILE static weights (recompile every call). Our solution: pack both operands into a **single IOSurface** along the spatial dimension:

```
IOSurface [1, in_ch, 1, spatial + out_ch]:
  sp[0:spatial]           = activations x[in_ch, spatial]
  sp[spatial:spatial+out] = weights W.T[in_ch, out_ch]
```

MIL: `slice_by_size → reshape → transpose → matmul → transpose → reshape`

Compile once per unique shape, update IOSurface data each call. **Zero recompile, zero handle leak.**

### 2. Fused 4-Matmul LoRA Gradient Kernel

All 4 gradient matmuls in a single MIL program with a single ANE dispatch:

```
Input:  [1, max_dim, 1, 2*ps + 2*pr]  (dy^T, x^T, B^T, A packed)
Output: d_A [1, in_dim, 1, pr] + d_B [1, pr, 1, out_dim]
```

**Benchmark (M4 Mini, 2048×2048, rank=8, seq=32):**
| Mode | Time/module | Dispatches |
|------|------------|------------|
| Fused | **0.36 ms** | 1 |
| Unfused | 0.72 ms | 4 |

### 3. `@{}` vs `nil` Weights Dict Fix

Passing `nil` for the weights parameter in `_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:` causes matmul/slice MIL programs to fail compilation. Passing `@{}` (empty dict) works. This is undocumented and critical for the dynamic matmul pattern.

### 4. ANE Spatial Dimension Constraint

The spatial (last) dimension must be **≥ 16 AND a multiple of 16**. Both input spatial and output channel dims are auto-padded. This applies to the matmul output dimensions too — small `out_ch` (like LoRA rank=8) must be padded to 16.

## Verified Results

**Hardware**: M4 Mac Mini, 16GB unified memory, macOS 15.3.1

**Model**: Qwen2.5-3B-Instruct-4bit (482M params, 446K trainable)

| Metric | Value |
|--------|-------|
| Training steps | 20 |
| Loss (start → end) | 4.07 → 2.85 |
| ANE dispatches per step | 8 (fused, 1 per module) |
| GPU fallbacks | 0 |
| Time per step | ~155 ms (steady state) |
| Fused kernel compile time | ~100 ms (once) |
| Numerical error vs numpy | < 0.001 |

## Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/jmanhype/ane-lora-training.git
cd ane-lora-training
pip install -r requirements.txt
cd bridge && make && cd ..
```

### 2. Run End-to-End Training

```bash
python examples/train_e2e.py
```

This loads Qwen2.5-3B-Instruct-4bit, applies LoRA to the last 2 layers, and runs 20 training steps with ANE fused gradient dispatch.

Environment variables:
- `ANE_MODEL` — HuggingFace model ID (default: `mlx-community/Qwen2.5-3B-Instruct-4bit`)
- `ANE_STEPS` — Training steps (default: `20`)
- `ANE_LORA_RANK` — LoRA rank (default: `8`)
- `ANE_LORA_LAYERS` — Last N layers to adapt (default: `2`)
- `ANE_LR` — Learning rate (default: `1e-4`)

### 3. Run the Live Fine-Tuning Daemon

Chat server with background ANE training — the model improves during conversation:

```bash
python examples/mlx_lora_daemon.py

# Chat with it
curl -s -X POST http://localhost:8766/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is LoRA?"}],"stream":false}'

# Check training stats
curl -s http://localhost:8766/status | python3 -m json.tool
```

### 4. Use the Kernel API Directly

```python
from ane_lora_kernels import ANELoRAKernels, set_ane_kernels, replace_lora_with_ane

# Initialize and verify
kernels = ANELoRAKernels("bridge/libane_bridge.dylib")
assert kernels.verify_conv() < 0.01

# Replace LoRA layers with ANE-backed versions
set_ane_kernels(kernels)
replaced = replace_lora_with_ane(model)

# Training now automatically routes LoRA gradients to ANE
loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens)
```

## Project Structure

```
ane-lora-training/
├── ane_lora_kernels.py              # Core: fused ANE dispatch + MLX VJP
├── bridge/
│   ├── ane_bridge.h                 # ANE bridge C header
│   ├── ane_bridge.m                 # ANE bridge Obj-C (private APIs)
│   └── Makefile                     # Build libane_bridge.dylib
├── examples/
│   ├── train_e2e.py                 # Standalone training (proves the stack)
│   └── mlx_lora_daemon.py           # Live chat + background ANE training
└── tests/
    ├── test_spatial_constraints.py
    ├── test_conv_matmul.py
    ├── test_gradient_pipeline.py
    └── test_subprocess_dispatch.py
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **OS**: macOS 15+ (Sequoia)
- **Xcode CLI Tools**: `xcode-select --install`
- **Python**: 3.11+ (tested on 3.14)
- **Dependencies**: `pip install -r requirements.txt`

## Numerical Accuracy

All computations use fp16 intermediate precision (ANE native):

| Operation | Max Error | Notes |
|-----------|-----------|-------|
| Dynamic matmul (small) | < 2e-4 | 64×64, spatial=32 |
| Dynamic matmul (large) | < 1.2e-3 | 2048×2048, spatial=32 |
| Fused LoRA gradient (q/o_proj) | < 1.6e-4 | dim=2048, rank=8 |
| Fused LoRA gradient (k/v_proj) | < 1.6e-4 | dim=2048→256, rank=8 |

## Dead Ends & What Didn't Work

1. **`matmul` with 2 dynamic inputs**: Compiles, eval always fails. Only works via packed-IOSurface with slice/reshape.
2. **Conv with dynamic weight inputs**: BLOBFILE static weights only. Dynamic weights via input tensor silently fail.
3. **Weight patching (disk/memory/API)**: ANE copies weights to SRAM at compile time. Only input-packing approaches work.
4. **`nil` for weights dict**: Causes compilation failure for matmul/slice MIL programs. Must pass `@{}`.
5. **Small spatial dims**: Spatial < 16 or not aligned to 16 → eval fails despite successful compilation.

## Related Work

- [maderix/ANE](https://github.com/maderix/ANE) — ANE reverse engineering, dynamic matmul PoC (ObjC)
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — LLM inference and LoRA on MLX

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{ane_lora_training,
  title={ANE LoRA Training: Fused Neural Engine Gradient Kernels for Apple Silicon},
  author={Straughter Guthrie},
  year={2026},
  url={https://github.com/jmanhype/ane-lora-training}
}
```
