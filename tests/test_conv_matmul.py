"""Test _conv_matmul with LoRA-relevant matrix shapes.

Verifies that conv-as-matmul produces correct results for every
shape encountered in LoRA gradient computation.

Requires: libane_bridge.dylib
"""
import ctypes
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_lora_kernels import _conv_matmul

BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/ANE-backup/bridge/libane_bridge.dylib"))


def setup_lib():
    """Initialize ANE bridge for direct _conv_matmul testing."""
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


def main():
    lib = setup_lib()
    if lib is None:
        sys.exit(1)

    np.random.seed(42)
    all_pass = True

    # LoRA gradient matmul shapes (rank=8, dim=2048, seq=32)
    test_cases = [
        # (W_shape, x_cols, description)
        # Step 1: B[rank,out] @ dy^T[out,seq] -> spatial=seq=32 (OK)
        ((8, 2048), 32, "Step1 q/o: B[8,2048] @ dy^T[2048,32]"),
        # Step 2: x^T[in,seq] @ tmp[seq,rank] -> spatial=rank=8 (auto-padded to 16)
        ((2048, 32), 8, "Step2 q/o: x^T[2048,32] @ tmp[32,8]"),
        # Step 3: A^T[rank,in] @ x^T[in,seq] -> spatial=seq=32 (OK)
        ((8, 2048), 32, "Step3 q/o: A^T[8,2048] @ x^T[2048,32]"),
        # Step 4: ax^T[rank,seq] @ dy[seq,out] -> spatial=out=2048 (OK)
        ((8, 32), 2048, "Step4 q/o: ax^T[8,32] @ dy[32,2048]"),
        # k/v_proj variants (out_dim=256)
        ((8, 2048), 32, "Step1 k/v: B[8,256] @ dy^T[256,32]"),
        ((2048, 32), 8, "Step2 k/v: x^T[2048,32] @ tmp[32,8]"),
        ((8, 256), 32, "Step4 k/v: ax^T[8,32] @ dy[32,256]"),
        # Larger spatial
        ((16, 16), 64, "Sanity: 16x16 sp=64"),
        ((64, 64), 32, "Sanity: 64x64 sp=32"),
        ((256, 256), 32, "Sanity: 256x256 sp=32"),
    ]

    print("=" * 60)
    print("Conv-as-MatMul Correctness Test")
    print("=" * 60)

    for W_shape, x_cols, desc in test_cases:
        W = np.random.randn(*W_shape).astype(np.float32) * 0.01
        x = np.random.randn(W_shape[1], x_cols).astype(np.float32)
        try:
            result = _conv_matmul(lib, W, x)
            expected = W @ x
            err = np.max(np.abs(result - expected))
            status = "PASS" if err < 0.01 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {desc}: {status} err={err:.6f}")
        except RuntimeError as e:
            all_pass = False
            print(f"  {desc}: EXCEPTION {e}")

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
