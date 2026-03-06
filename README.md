# ANE LoRA Training: Neural Engine Gradient Computation on Apple Silicon

**First public demonstration of LoRA fine-tuning gradients running on Apple's Neural Engine (ANE) via private APIs.**

LoRA gradient computation — the 4 matrix multiplications per adapter module — dispatched to ANE at ~2.8W, while MLX handles inference on the GPU. Verified on M4 Mac Mini with Qwen2.5-3B-Instruct-4bit, 16 LoRA modules, converging loss across training steps.

```
[ANE-FT] step=1 loss=3.3594 | ane_dispatches=64 fallbacks=0 | 2.7s
[ANE-FT] step=2 loss=3.3516 | ane_dispatches=64 fallbacks=0 | 2.7s
[ANE-FT] step=3 loss=3.3398 | ane_dispatches=64 fallbacks=0 | 2.6s
[ANE-FT] step=4 loss=3.3223 | ane_dispatches=64 fallbacks=0 | 2.6s
[ANE-FT] step=5 loss=3.3008 | ane_dispatches=64 fallbacks=0 | 2.5s
```

## Key Discoveries

### 1. `matmul` MIL Op Does NOT Work on ANE

The MIL `matmul` operation **compiles successfully** but **always fails at eval time**. This is not documented anywhere. Only `conv` works for compute on the Neural Engine.

All matrix multiplications must be expressed as 1x1 convolutions:

```
A[M,K] @ B[K,N]  →  conv(input=[1,K,1,N], weight=[M,K,1,1]) → [1,M,1,N]
```

### 2. ANE Spatial Dimension Constraint

The spatial (last) dimension of conv inputs must be **>= 16 AND a multiple of 16**. Kernels that violate this compile fine but fail at eval.

| Spatial | Works? | Why |
|---------|--------|-----|
| 4       | NO     | Below minimum (16) |
| 8       | NO     | Below minimum — this is LoRA rank! |
| 12      | NO     | Below minimum |
| 16      | YES    | Minimum viable |
| 24      | NO     | Not aligned to 16 |
| 32      | YES    | Aligned |
| 2048    | YES    | Large values are fine |

**Channel ratios are irrelevant** — even 8→2048 works when spatial >= 16. The earlier hypothesis (from the maderix/ANE project) that asymmetric channels caused failures was wrong. It was always the spatial dimension.

### 3. Conv-as-MatMul for LoRA Gradients

Each LoRA module requires 4 gradient matmuls. We express each as a conv operation:

```python
# LoRA: y = (x @ lora_a @ lora_b) * scale
# Gradients needed: d_lora_a and d_lora_b

# Step 1: tmp = dy @ B^T  →  conv(W=B[rank,out], X=dy^T[out,seq])^T
tmp = conv_matmul(lora_b, dy.T).T           # [seq, rank]

# Step 2: d_A = x^T @ tmp  →  conv(W=x^T[in,seq], X=tmp[seq,rank])
d_lora_a = conv_matmul(x.T, tmp)            # [in, rank]
# ↑ spatial=rank=8, auto-padded to 16

# Step 3: ax = x @ A  →  conv(W=A^T[rank,in], X=x^T[in,seq])^T
ax = conv_matmul(lora_a.T, x.T).T           # [seq, rank]

# Step 4: d_B = ax^T @ dy  →  conv(W=ax^T[rank,seq], X=dy[seq,out])
d_lora_b = conv_matmul(ax.T, dy)            # [rank, out]
```

### 4. Subprocess Isolation for Compile Budget

ANE leaks ~1 compile handle per `ane_bridge_compile()` call. After ~119 compiles, new compiles silently fail. Solution: run each training step's gradient computation in a subprocess with a fresh ANE bridge. 64 compiles per step (16 modules × 4 steps) fits within budget.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     MLX Inference (GPU)                       │
│                                                              │
│  model.generate()  →  LoRA forward:  base(x) + scale*(x@A@B)│
│                                                              │
│  loss = cross_entropy(pred, target)                          │
│  grads = nn.value_and_grad(loss_fn)(model, ...)              │
│                                                              │
│  mx.custom_function VJP intercepts LoRA backward pass:       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  dx → stays on GPU (chain rule for earlier layers)     │  │
│  │  d_lora_a, d_lora_b → dispatched to ANE subprocess ──────►│
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                                                        │
                                    ┌───────────────────▼──────┐
                                    │   ANE Subprocess (~2.8W) │
                                    │                          │
                                    │  for each LoRA module:   │
                                    │    4x conv kernel compile│
                                    │    4x ANE eval dispatch  │
                                    │    kernel free           │
                                    │                          │
                                    │  Returns: d_A, d_B numpy │
                                    └──────────────────────────┘
```

## Project Structure

```
ane-lora-training/
├── ane_lora_kernels.py           # Core: conv-based ANE dispatch + MLX VJP
├── tests/
│   ├── test_spatial_constraints.py  # ANE dimension constraint discovery
│   ├── test_conv_matmul.py          # Direct conv-as-matmul verification
│   ├── test_gradient_pipeline.py    # Full 4-step gradient pipeline
│   └── test_subprocess_dispatch.py  # End-to-end subprocess dispatch
├── examples/
│   └── mlx_lora_daemon.py           # Full daemon: MLX inference + ANE training
└── docs/
    └── findings.md                  # Detailed technical findings & dead ends
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) — tested on M4 Mac Mini 16GB
- **OS**: macOS 15+ (Sequoia)
- **ANE Bridge**: `libane_bridge.dylib` from [maderix/ANE](https://github.com/maderix/ANE)
- **Python**: 3.11+ (tested on 3.14)
- **Dependencies**: `numpy` (core), `mlx`, `mlx-lm` (for MLX integration)

## Quick Start

### 1. Build the ANE Bridge

```bash
git clone https://github.com/maderix/ANE.git
cd ANE/bridge
clang -framework Foundation -framework IOKit -framework CoreML \
  -dynamiclib -o libane_bridge.dylib ane_bridge.m
```

### 2. Run the Tests

```bash
# Set your bridge path
export ANE_BRIDGE_PATH=/path/to/libane_bridge.dylib

# Test spatial constraints (understand ANE limits)
python tests/test_spatial_constraints.py

# Test conv-as-matmul correctness
python tests/test_conv_matmul.py

# Test full gradient pipeline
python tests/test_gradient_pipeline.py

# Test subprocess dispatch (end-to-end)
python tests/test_subprocess_dispatch.py
```

### 3. Use in Your MLX Training Loop

```python
from ane_lora_kernels import ANELoRAKernels

# Initialize
kernels = ANELoRAKernels("/path/to/libane_bridge.dylib")

# Verify ANE works
err = kernels.verify_conv()
assert err < 0.01, f"ANE conv error: {err}"

# Compute gradients for LoRA modules
# modules = [(dy, x, lora_a, lora_b), ...]  # numpy arrays
results = kernels.compute_lora_gradients(modules)
for d_lora_a, d_lora_b in results:
    # Apply gradients...
```

### 4. Full MLX Integration

```python
from ane_lora_kernels import ANELoRAKernels, set_ane_kernels, replace_lora_with_ane

# Load model with LoRA
model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
# ... apply LoRA adapters ...

# Initialize ANE and replace LoRA layers
kernels = ANELoRAKernels(bridge_path)
set_ane_kernels(kernels)
n = replace_lora_with_ane(model)
print(f"Replaced {n} LoRA layers with ANE-backed versions")

# Training now automatically dispatches LoRA gradients to ANE
loss, grads = nn.value_and_grad(model, loss_fn)(model, ...)
```

## MIL (Model Intermediate Language) Reference

### Correct Conv MIL Syntax

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"},
{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""},
{"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, IN_CH, 1, SPATIAL]> x) {
        // Constants
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];

        // Cast input to fp16
        tensor<fp16, [1, IN_CH, 1, SPATIAL]> x16 = cast(
            dtype = to_fp16, x = x
        )[name = string("cast_in")];

        // Weight from blob file (fp16, baked at compile time)
        tensor<fp16, [OUT_CH, IN_CH, 1, 1]> W = const()[
            name = string("W"),
            val = tensor<fp16, [OUT_CH, IN_CH, 1, 1]>(BLOBFILE(
                path = string("@model_path/weights/weight.bin"),
                offset = uint64(64)
            ))
        ];

        // Conv (the only compute op that works on ANE!)
        tensor<fp16, [1, OUT_CH, 1, SPATIAL]> y16 = conv(
            x = x16, weight = W,
            pad_type = string("valid"), strides = [1, 1]
        )[name = string("conv")];

        // Cast output back to fp32
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, OUT_CH, 1, SPATIAL]> y = cast(
            dtype = to_fp32, x = y16
        )[name = string("cast_out")];
    } -> (y);
}
```

### Weight Blob Format

128-byte header (64 global + 64 chunk) followed by fp16 weight data:

```
Offset  Size  Field
0x00    4     Version (1)
0x04    4     Type (2 = fp16)
0x08    56    Reserved zeros
0x40    4     Magic (0xDEADBEEF, little-endian)
0x44    4     Chunk count (1)
0x48    4     Data size (fp16 byte count)
0x4C    4     Reserved
0x50    4     Data offset (128)
0x54    12    Reserved zeros
0x80    ...   fp16 weight data (row-major)
```

## Numerical Accuracy

All computations use fp16 intermediate precision (ANE native). Expected errors vs fp32 numpy:

| Operation | Max Error | Notes |
|-----------|-----------|-------|
| Single conv matmul (small) | < 1e-4 | 16x16, spatial=32 |
| Single conv matmul (large) | < 1e-3 | 2048x8, spatial=32 |
| Full LoRA gradient (q/o_proj) | < 1e-3 | dim=2048, rank=8 |
| Full LoRA gradient (k/v_proj) | < 5e-4 | dim=256, rank=8 |
| 16-module subprocess dispatch | < 2e-4 | All modules combined |

## Verified Results

**Hardware**: M4 Mac Mini, 16GB unified memory, macOS 15.3.1

**Model**: Qwen2.5-3B-Instruct-4bit (QuantizedLinear)

**LoRA Config**: rank=8, last 4 transformer layers, q/k/v/o projections (16 modules)

| Metric | Value |
|--------|-------|
| Training steps | 5 |
| Loss (start → end) | 3.3594 → 3.3008 |
| ANE dispatches per step | 64 (16 modules × 4 matmuls) |
| GPU fallbacks | 0 |
| Time per step | ~2.7s |
| Total ANE dispatches (3 chats) | 192 |
| Max numerical error | 0.000192 |

## Dead Ends & What Didn't Work

Documenting these to save others time:

1. **`matmul` MIL op**: Compiles, eval always returns false. Spent hours debugging before discovering only `conv` works.

2. **Power-of-2 spatial padding**: Originally padded to 8, 16, 32, 64... but 8 doesn't work. The actual constraint is multiples of 16, minimum 16.

3. **Channel ratio hypothesis**: Thought asymmetric channels (8:2048) caused eval failures. Wrong — it was spatial < 16 in every case.

4. **`multiprocessing.spawn`**: Unreliable for ANE subprocess isolation. `subprocess.run` with inline `-c` script is simpler and works.

5. **`os.path.dirname(os.path.abspath(__file__))` in subprocess**: Doesn't work in `-c` scripts since `__file__` isn't defined. Pre-compute the path in the parent process.

6. **`model.load_weights(updates)` with partial params**: Fails with "Missing N parameters". Must use `strict=False`.

7. **Mixed dtype VJP**: `mx.custom_function` VJP must return tangents matching primal dtypes exactly, or you get `ValueError: Type of cotangents does not match primal output type`.

## How It Works (Deep Dive)

See [docs/findings.md](docs/findings.md) for the complete technical narrative including:
- ANE private API calling conventions via ctypes
- MIL syntax rules (program 1.3, func main<ios18>)
- Weight blob reverse engineering
- Conv kernel compile/eval/free lifecycle
- IOSurface I/O format (fp32 in, internal fp16, fp32 out)
- Compile handle leak measurement and subprocess mitigation
- MLX `mx.custom_function` VJP integration pattern

## Related Work

- [maderix/ANE](https://github.com/maderix/ANE) — ANE bridge and MIL generation (inference only)
- [maderix Substack: Inside the M4 Apple Neural Engine](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) — Reverse engineering ANE private APIs
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — LLM inference and LoRA fine-tuning on MLX

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this work, please cite:

```bibtex
@software{ane_lora_training,
  title={ANE LoRA Training: Neural Engine Gradient Computation on Apple Silicon},
  author={Ex0byt},
  year={2026},
  url={https://github.com/maderix/ane-lora-training}
}
```
