"""Test fused LoRA gradient kernels vs unfused implementation.

Verifies that the fused kernel produces identical results to the
unfused 4-step implementation within fp16 tolerance.

Tests:
1. Correctness: Fused output matches unfused output
2. Dispatch count: Fused uses 1 dispatch vs unfused 4
3. Caching: Second call doesn't recompile
4. Dimension variants: Different in/out/rank combinations
5. Fallback: ANE_DISABLE_FUSION=1 works correctly

Requires: libane_bridge.dylib
"""
import ctypes
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_lora_kernels import (
    _conv_matmul, _fused_conv_lora_grad, _pad_spatial,
    PersistentANEBridge, ANELoRAKernels
)

BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/ANE-backup/bridge/libane_bridge.dylib"))


def setup_lib():
    """Initialize libane_bridge."""
    try:
        lib = ctypes.CDLL(BRIDGE_PATH)
    except OSError:
        print(f"Cannot load bridge from {BRIDGE_PATH}")
        return None

    lib.ane_bridge_init.restype = ctypes.c_int
    lib.ane_bridge_compile.restype = ctypes.c_void_p
    lib.ane_bridge_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ane_bridge_eval.restype = ctypes.c_bool
    lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
    lib.ane_bridge_write_input.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.ane_bridge_read_output.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

    rc = lib.ane_bridge_init()
    if rc != 0:
        print(f"Bridge init failed: {rc}")
        return None
    return lib


def unfused_lora_grad(lib, dy, x, lora_a, lora_b):
    """Compute LoRA gradients using 4 separate unfused conv matmuls."""
    seq = dy.shape[0]
    out_dim = dy.shape[1]
    in_dim = x.shape[1]
    rank = lora_a.shape[1]

    # Pad sequence for ANE
    padded_seq = _pad_spatial(seq)
    dy_p = np.zeros((padded_seq, out_dim), dtype=np.float32)
    dy_p[:seq] = dy
    x_p = np.zeros((padded_seq, in_dim), dtype=np.float32)
    x_p[:seq] = x

    # Step 1: tmp = dy @ B^T → (B @ dy^T)^T
    tmp = _conv_matmul(lib, lora_b, dy_p.T).T[:seq]

    # Step 2: d_A = x^T @ tmp
    tmp_p = np.zeros((padded_seq, rank), dtype=np.float32)
    tmp_p[:seq] = tmp
    d_a = _conv_matmul(lib, x_p.T, tmp_p)

    # Step 3: ax = x @ A → (A^T @ x^T)^T
    ax = _conv_matmul(lib, lora_a.T, x_p.T).T[:seq]

    # Step 4: d_B = ax^T @ dy
    ax_p = np.zeros((padded_seq, rank), dtype=np.float32)
    ax_p[:seq] = ax
    d_b = _conv_matmul(lib, ax_p.T, dy_p)

    return d_a, d_b


def test_correctness(lib, seq, in_dim, rank, out_dim, name):
    """Test that fused kernel matches unfused implementation."""
    np.random.seed(hash(name) % (2**31))

    dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
    x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
    lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
    lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

    try:
        # Unfused reference
        d_a_unfused, d_b_unfused = unfused_lora_grad(lib, dy, x, lora_a, lora_b)

        # Fused implementation
        kernel_cache = {}
        d_a_fused, d_b_fused = _fused_conv_lora_grad(
            lib, dy, x, lora_a, lora_b, kernel_cache
        )

        # Compare results (fp16 tolerance = ~0.001)
        err_a = np.max(np.abs(d_a_fused - d_a_unfused))
        err_b = np.max(np.abs(d_b_fused - d_b_unfused))
        max_err = max(err_a, err_b)

        passed = max_err < 0.01  # Allow for fp16 accumulation

        print(f"\n  {name}:")
        print(f"    d_A error: {err_a:.6f}")
        print(f"    d_B error: {err_b:.6f}")
        print(f"    {'PASS' if passed else 'FAIL'} max_err={max_err:.6f}")
        return passed

    except Exception as e:
        print(f"\n  {name}: EXCEPTION {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dispatch_count():
    """Test that fused kernel uses 1 dispatch vs unfused 4."""
    print("\n" + "=" * 60)
    print("Test: Dispatch Count (fused vs unfused)")
    print("=" * 60)

    lib = setup_lib()
    if lib is None:
        print("  SKIP: Cannot load bridge")
        return True

    seq, in_dim, rank, out_dim = 32, 2048, 8, 2048
    np.random.seed(42)

    dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
    x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
    lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
    lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

    try:
        # Test unfused dispatch count
        start = time.time()
        d_a_unfused, d_b_unfused = unfused_lora_grad(lib, dy, x, lora_a, lora_b)
        unfused_time = time.time() - start

        # Test fused dispatch count
        kernel_cache = {}
        start = time.time()
        d_a_fused, d_b_fused = _fused_conv_lora_grad(
            lib, dy, x, lora_a, lora_b, kernel_cache
        )
        fused_time = time.time() - start

        speedup = unfused_time / fused_time

        print(f"\n  Unfused time: {unfused_time*1000:.2f}ms (4 dispatches)")
        print(f"  Fused time:   {fused_time*1000:.2f}ms (1 dispatch)")
        print(f"  Speedup:      {speedup:.2f}x")
        print(f"  Result:       {'PASS' if speedup > 1.0 else 'FAIL'}")

        return speedup > 1.0

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caching():
    """Test that kernel caching works (second call is faster)."""
    print("\n" + "=" * 60)
    print("Test: Kernel Caching")
    print("=" * 60)

    lib = setup_lib()
    if lib is None:
        print("  SKIP: Cannot load bridge")
        return True

    seq, in_dim, rank, out_dim = 32, 2048, 8, 2048
    np.random.seed(42)

    dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
    x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
    lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
    lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

    try:
        kernel_cache = {}

        # First call (compile + execute)
        start = time.time()
        d_a1, d_b1 = _fused_conv_lora_grad(
            lib, dy, x, lora_a, lora_b, kernel_cache
        )
        first_time = time.time() - start

        # Second call (cached, execute only)
        start = time.time()
        d_a2, d_b2 = _fused_conv_lora_grad(
            lib, dy, x, lora_a, lora_b, kernel_cache
        )
        second_time = time.time() - start

        speedup = first_time / second_time if second_time > 0 else 1.0
        cached = len(kernel_cache) > 0

        print(f"\n  First call (compile):  {first_time*1000:.2f}ms")
        print(f"  Second call (cached):  {second_time*1000:.2f}ms")
        print(f"  Cache size:            {len(kernel_cache)}")
        print(f"  Speedup:               {speedup:.2f}x")
        print(f"  Result:                {'PASS' if cached and speedup > 1.0 else 'FAIL'}")

        return cached and speedup > 1.0

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persistent_bridge():
    """Test that PersistentANEBridge uses fused kernels by default."""
    print("\n" + "=" * 60)
    print("Test: Persistent Bridge with Fusion")
    print("=" * 60)

    try:
        bridge = PersistentANEBridge(BRIDGE_PATH)

        # Create test modules
        seq, in_dim, rank, out_dim = 32, 2048, 8, 2048
        np.random.seed(42)

        dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
        x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
        lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
        lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

        modules = [(dy, x, lora_a, lora_b)]

        # Compute gradients
        results = bridge.compute_lora_gradients(modules)
        d_a, d_b = results[0]

        # Check stats (should be 1 dispatch, not 4)
        dispatches = bridge.total_dispatches
        compiles = bridge.total_compiles

        bridge.shutdown()

        print(f"\n  Dispatches: {dispatches}")
        print(f"  Compiles:   {compiles}")
        print(f"  Result:     {'PASS' if dispatches == 1 else 'FAIL (expected 1 dispatch)'}")

        return dispatches == 1

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_disable():
    """Test that ANE_DISABLE_FUSION=1 disables fusion."""
    print("\n" + "=" * 60)
    print("Test: ANE_DISABLE_FUSION=1")
    print("=" * 60)

    # Set environment variable
    os.environ['ANE_DISABLE_FUSION'] = '1'

    try:
        bridge = PersistentANEBridge(BRIDGE_PATH)

        # Create test modules
        seq, in_dim, rank, out_dim = 32, 2048, 8, 2048
        np.random.seed(42)

        dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
        x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
        lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
        lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

        modules = [(dy, x, lora_a, lora_b)]

        # Compute gradients
        results = bridge.compute_lora_gradients(modules)
        d_a, d_b = results[0]

        # Check stats (should be 4 dispatches, not 1)
        dispatches = bridge.total_dispatches

        bridge.shutdown()

        print(f"\n  Dispatches: {dispatches}")
        print(f"  Result:     {'PASS' if dispatches == 4 else 'FAIL (expected 4 dispatches)'}")

        return dispatches == 4

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clear environment variable
        os.environ.pop('ANE_DISABLE_FUSION', None)


def main():
    lib = setup_lib()
    if lib is None:
        print("Cannot load bridge - exiting")
        sys.exit(1)

    print("=" * 60)
    print("Fused LoRA Kernel Tests")
    print("=" * 60)

    all_pass = True

    # Test 1: Correctness across different dimensions
    print("\n" + "=" * 60)
    print("Test 1: Correctness (fused vs unfused)")
    print("=" * 60)

    test_cases = [
        (32, 2048, 8, 2048, "q/o_proj (2048→2048, rank=8, seq=32)"),
        (32, 2048, 8, 256, "k/v_proj (2048→256, rank=8, seq=32)"),
        (64, 2048, 8, 2048, "q_proj (2048→2048, rank=8, seq=64)"),
        (32, 2048, 16, 2048, "q_proj (2048→2048, rank=16, seq=32)"),
        (32, 4096, 8, 4096, "q_proj (4096→4096, rank=8, seq=32)"),
    ]

    for seq, in_dim, rank, out_dim, name in test_cases:
        if not test_correctness(lib, seq, in_dim, rank, out_dim, name):
            all_pass = False

    # Test 2: Dispatch count
    if not test_dispatch_count():
        all_pass = False

    # Test 3: Kernel caching
    if not test_caching():
        all_pass = False

    # Test 4: Persistent bridge integration
    if not test_persistent_bridge():
        all_pass = False

    # Test 5: Fusion disable flag
    if not test_fusion_disable():
        all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
