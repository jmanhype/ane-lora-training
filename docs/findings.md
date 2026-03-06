# Technical Findings: ANE LoRA Gradient Dispatch

Detailed record of discoveries, dead ends, and solutions from getting LoRA gradient
computation running on Apple's Neural Engine via private APIs.

## 1. ANE Private API Calling Convention

The ANE is accessed via `libane_bridge.dylib`, a C wrapper around Apple's private
`ANECompiler` and `ANEDevice` frameworks. The bridge exposes:

```c
int    ane_bridge_init(void);
void*  ane_bridge_compile(const char* mil, size_t mil_len,
                          const uint8_t* weights, size_t weights_len,
                          int n_inputs, const size_t* input_sizes,
                          int n_outputs, const size_t* output_sizes);
bool   ane_bridge_eval(void* kernel);
void   ane_bridge_write_input(void* kernel, int idx, const void* data, size_t len);
void   ane_bridge_read_output(void* kernel, int idx, void* data, size_t len);
void   ane_bridge_free(void* kernel);
int    ane_bridge_get_compile_count(void);
```

Python access is via `ctypes.CDLL`. All data transfer uses fp32 IOSurfaces.
The ANE handles fp16 conversion internally based on MIL cast operations.

## 2. MIL (Model Intermediate Language) Syntax

### What Works

The ANE compiler accepts `program(1.3)` format with `func main<ios18>` entry point.
Key syntax rules discovered through trial and error:

- **String literals**: `string("value")` not `"value"`
- **Bool literals**: `bool(false)` not `false`
- **No `%` prefix**: Variable names are bare identifiers
- **Typed declarations**: `tensor<fp32, [1, 8, 1, 32]> x` (no `=` in function params)
- **Const pattern**: `type name = const()[name = string("..."), val = ...]`
- **Op attributes**: `[name = string("...")]` suffix on each operation

### buildInfo Metadata

The preamble `[buildInfo = dict<string, string>({...})]` appears required.
Omitting it may cause silent compilation failures. We match coremltools 9.0 output:

```
[buildInfo = dict<string, string>({
  {"coremlc-component-MIL", "3510.2.1"},
  {"coremlc-version", "3505.4.1"},
  {"coremltools-component-milinternal", ""},
  {"coremltools-version", "9.0"}
})]
```

### What Does NOT Work

- **`matmul` MIL op**: Compiles successfully but `ane_bridge_eval()` always returns false.
  This was the biggest surprise. Spent hours debugging MIL syntax before discovering
  the fundamental limitation. Only `conv` works for compute operations.

- **Pseudo-MIL syntax**: `%var = op(...)` format seen in some documentation doesn't work.
  The real format uses typed variable declarations.

## 3. Spatial Dimension Constraint

### Discovery Process

Initial testing showed asymmetric channel dims (8×2048) failing, leading to the
hypothesis that extreme channel ratios caused problems. Systematic testing
(`test_spatial_constraints.py`) revealed the real constraint:

**Spatial (last dimension of conv input) must be >= 16 AND a multiple of 16.**

| Spatial | Compile | Eval   | Why |
|---------|---------|--------|-----|
| 4       | OK      | FAIL   | Below minimum |
| 8       | OK      | FAIL   | Below minimum (this is LoRA rank!) |
| 12      | OK      | FAIL   | Below minimum |
| 16      | OK      | PASS   | Minimum viable dimension |
| 24      | OK      | FAIL   | Not aligned to 16 |
| 32      | OK      | PASS   | Aligned |
| 64      | OK      | PASS   | Aligned |
| 2048    | OK      | PASS   | Large values fine |

### Impact on LoRA

For LoRA with rank=8, Step 2 of gradient computation has spatial=rank=8.
This needs to be padded to 16. The `_pad_spatial()` function handles this
automatically in `_conv_matmul()` — pads input with zeros, truncates output.

### Channel Ratios Are Irrelevant

With spatial=32 (a valid dimension), ALL channel ratios work:
- 8→8: PASS
- 8→2048: PASS
- 2048→8: PASS
- 8→16384: PASS

The earlier maderix/ANE project notes about "asymmetric dims failing" were
actually spatial constraint violations, not channel ratio issues.

## 4. Weight Blob Format

The `BLOBFILE()` MIL reference points to a binary blob with a specific header format.
Reverse-engineered from working models:

```
Offset 0x00 (64 bytes): Global header
  [0:4]   = 0x00000001  (version 1)
  [4:8]   = 0x00000002  (type fp16)
  [8:64]  = zeros

Offset 0x40 (64 bytes): Chunk header
  [0:4]   = 0xDEADBEEF  (magic, little-endian)
  [4:8]   = 0x00000001  (chunk count)
  [8:12]  = data_size   (fp16 byte count)
  [12:16] = 0           (reserved)
  [16:20] = 128         (data offset from file start)
  [20:64] = zeros

Offset 0x80: fp16 weight data (row-major)
```

The MIL references this with `offset = uint64(64)` which points to the chunk
header. The ANE compiler then reads data_offset from the chunk header to find
the actual fp16 data at offset 128.

## 5. Conv-as-MatMul Technique

### Core Idea

Matrix multiplication `A[M,K] @ B[K,N]` is equivalent to:
```
conv(input=[1,K,1,N], weight=[M,K,1,1]) → [1,M,1,N]
```

This works because 1×1 convolution with no padding is mathematically identical
to matrix multiplication across the spatial dimension.

### Operand Assignment for LoRA

Each LoRA gradient requires 4 matmuls. We assign operands to minimize recompilation:

| Step | Matmul | Conv Weight (baked) | Conv Input (IOSurface) |
|------|--------|-------------------|----------------------|
| 1 | dy @ B^T | lora_b[r,out] | dy^T[out,seq] |
| 2 | x^T @ tmp | x^T[in,seq] | tmp[seq,r] |
| 3 | x @ A | lora_a^T[r,in] | x^T[in,seq] |
| 4 | ax^T @ dy | ax^T[r,seq] | dy[seq,out] |

Since weights change every training step, we compile fresh kernels per call.
The compile cost (~1ms per kernel) is negligible compared to data transfer.

### Transpose Handling

To compute `A @ B^T` we use `conv(W=A, X=B^T)` — the transpose is done on
the CPU (numpy `.T`) before writing to IOSurface. The conv output may also
need transposing depending on which way the result is oriented.

## 6. Compile Handle Leak

### Measurement

Each `ane_bridge_compile()` call leaks a handle. After ~119 calls in a single
process, new compiles silently return NULL. The bridge provides
`ane_bridge_get_compile_count()` to track this.

### Subprocess Isolation

We run each training step's gradient computation in a fresh subprocess:
```python
proc = subprocess.run([sys.executable, "-c", script], timeout=30, ...)
```

Each subprocess gets a fresh ~119 compile budget. With 16 LoRA modules × 4 steps
= 64 compiles per training step, we have ~55 compiles of headroom.

Data exchange via numpy files in a temp directory:
- Parent writes `m{i}_dy.npy`, `m{i}_x.npy`, `m{i}_a.npy`, `m{i}_b.npy`
- Subprocess writes `m{i}_d_a.npy`, `m{i}_d_b.npy`, `stats.npy`

### Why Not multiprocessing.spawn?

We tried `multiprocessing.Process(target=...)` with spawn context but found
`subprocess.run` with inline `-c` script more reliable. Key gotcha:
`os.path.dirname(os.path.abspath(__file__))` doesn't work in `-c` scripts
since `__file__` is undefined. Solution: pre-compute the path in the parent
and inject via f-string with `{module_dir!r}`.

## 7. MLX Integration

### Custom VJP Pattern

MLX's `mx.custom_function` lets us intercept the backward pass of the LoRA
component while leaving the base layer (QuantizedLinear) untouched:

```python
@mx.custom_function
def _ane_lora_part(x, lora_a, lora_b, scale_arr):
    # Forward: scale * (x @ A) @ B
    return (scale_arr * (x.astype(mx.float32) @ lora_a) @ lora_b).astype(x.dtype)

@_ane_lora_part.vjp
def vjp(primals, cotangent, output):
    # dx on GPU, d_lora_a/d_lora_b on ANE
    ...
```

### QuantizedLinear Compatibility

4-bit quantized models use `nn.QuantizedLinear` instead of `nn.Linear`.
The ANE wrapper must:
1. Not assume `nn.Linear` — check `not isinstance(orig, LoRALinear)` instead
2. Access `lora_linear.linear` for the base layer (may be QuantizedLinear)
3. Call `self.linear(x)` for base forward (works for both Linear and QuantizedLinear)

### dtype Matching

`mx.custom_function` VJP must return tangents with dtypes matching primals exactly.
Quantized models often use float16 inputs. Our solution:
- Promote to float32 in forward for computation
- Cast output to `x.dtype` before returning
- Cast each tangent to its corresponding primal's dtype in VJP

### SGD Update with Partial Parameters

`model.load_weights(updates)` fails when updates only contain trainable params
(LoRA weights) but the model has many frozen params. Fix:
```python
model.load_weights(updates, strict=False)
```

## 8. Numerical Precision

ANE operates in fp16 internally (per the MIL cast operations). Expected errors
vs fp32 numpy reference:

| Scale | Typical Error | Max Observed |
|-------|--------------|-------------|
| Small matrices (16×16) | ~1e-5 | 1e-4 |
| Medium matrices (256×8) | ~1e-4 | 5e-4 |
| Large matrices (2048×8) | ~1e-3 | 2e-3 |
| Full 16-module pipeline | ~1e-4 | 2e-4 |

Errors > 0.01 indicate a bug (wrong transpose, wrong padding, etc.).
The fp16 intermediate precision is sufficient for LoRA training — loss
converges monotonically with ANE gradients.

## 9. Performance

Measured on M4 Mac Mini (16GB unified memory), macOS 15.3.1:

| Metric | Value |
|--------|-------|
| Single conv matmul | ~0.5ms |
| Subprocess startup | ~200ms |
| Full 16-module gradient | ~2.7s |
| Compile per kernel | ~1ms |
| IOSurface read/write | ~0.1ms |
| Numpy file I/O overhead | ~500ms |

The subprocess startup and numpy file I/O dominate. Future optimization:
shared memory or mmap instead of numpy files, persistent subprocess with
fresh bridge (via exec()), or direct ctypes dispatch with compile budget
management.
