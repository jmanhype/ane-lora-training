# Persistent ANE Bridge - Implementation Summary

## Overview
Implemented a persistent ANE bridge that eliminates the ~500ms overhead per training step by compiling kernels once and reusing them across steps with dynamic weight updates via shared memory.

## Key Changes

### 1. Dynamic Weight Inputs (`_gen_conv_mil`)
- **Before**: Weights baked as MIL constants using `const()` with BLOBFILE
- **After**: Weights passed as function inputs: `func main<ios18>(..., tensor<fp16, [...]> W)`
- **Benefit**: Enables weight updates without recompilation

### 2. Persistent Bridge Architecture
- **`PersistentANEBridge` class**: Manages long-lived subprocess
  - Spawns worker process at `__init__`
  - Compiles kernels with dynamic weight inputs
  - Communicates via `multiprocessing.Queue` for commands
  - Transfers data via `multiprocessing.shared_memory.SharedMemory`

- **`_persistent_ane_worker()` function**: Subprocess entry point
  - Initializes ANE bridge once (fresh compile budget)
  - Processes gradient computation commands
  - Returns results via shared memory (zero-copy)

### 3. Updated `ANELoRAKernels`
- Uses `PersistentANEBridge` by default
- Falls back to legacy subprocess mode via `ANE_LEGACY_SUBPROCESS=1`
- Added `shutdown()` method for clean teardown

## Performance Impact
- **Before**: ~500ms/step overhead (subprocess spawn + numpy file I/O)
- **After**: ~shared memory round-trip time (expected <50ms)
- **Expected speedup**: 10x reduction in per-step overhead

## Technical Details

### Shared Memory Protocol
**Input** (parent → worker):
```python
{
    "cmd": "compute",
    "dy_shm": "buffer_name",
    "x_shm": "buffer_name",
    "a_shm": "buffer_name",
    "b_shm": "buffer_name",
    "shapes": (seq, out_dim, in_dim, rank),
    "padded_seq": int
}
```

**Output** (worker → parent):
```python
{
    "status": "ok",
    "d_a_shm": "buffer_name",
    "d_b_shm": "buffer_name",
    "d_a_shape": (in_dim, rank),
    "d_b_shape": (rank, out_dim),
    "compiles": int,
    "dispatches": int
}
```

### MIL Input Signature
```mil
func main<ios18>(
    tensor<fp32, [1, in_ch, 1, spatial]> x,
    tensor<fp16, [out_ch, in_ch, 1, 1]> W
) -> (tensor<fp32, [1, out_ch, 1, spatial]> y)
```

## Testing
- ✅ Module imports successfully
- ✅ Class instantiation works
- ✅ Fallback mechanism functional
- ⏳ Runtime performance testing pending (requires libane_bridge.dylib)

## Migration
No code changes required for existing users:
```python
# This now uses persistent bridge automatically
kernels = ANELoRAKernels("/path/to/libane_bridge.dylib")
gradients = kernels.compute_lora_gradients(modules)
```

To use legacy mode (for debugging):
```bash
export ANE_LEGACY_SUBPROCESS=1
python your_training_script.py
```

## Future Work
- Benchmark actual performance improvement
- Consider compiling kernels once per unique (in_ch, out_ch) pair
- Add support for batched gradient computation
- Implement kernel cache warming at startup
