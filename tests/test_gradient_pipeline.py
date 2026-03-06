"""Test full LoRA gradient pipeline: 4 conv matmul steps vs numpy reference.

Verifies end-to-end gradient correctness for both q/o_proj (dim=2048)
and k/v_proj (dim=256) variants.

Requires: libane_bridge.dylib
"""
import ctypes
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_lora_kernels import _conv_matmul, _pad_spatial

BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/ANE-backup/bridge/libane_bridge.dylib"))


def setup_lib():
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


def test_gradient(lib, seq, in_dim, rank, out_dim, name):
    """Test full 4-step LoRA gradient computation."""
    np.random.seed(hash(name) % (2**31))

    dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
    x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
    lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
    lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

    # Numpy reference
    tmp_ref = dy @ lora_b.T          # [seq, rank]
    d_a_ref = x.T @ tmp_ref          # [in, rank]
    ax_ref = x @ lora_a              # [seq, rank]
    d_b_ref = ax_ref.T @ dy          # [rank, out]

    # Pad sequence for ANE
    padded_seq = _pad_spatial(seq)
    dy_p = np.zeros((padded_seq, out_dim), dtype=np.float32)
    dy_p[:seq] = dy
    x_p = np.zeros((padded_seq, in_dim), dtype=np.float32)
    x_p[:seq] = x

    try:
        # Step 1: tmp = dy @ B^T → (B @ dy^T)^T
        tmp = _conv_matmul(lib, lora_b, dy_p.T).T[:seq]
        err1 = np.max(np.abs(tmp - tmp_ref))

        # Step 2: d_A = x^T @ tmp (spatial=rank, auto-padded)
        tmp_p = np.zeros((padded_seq, rank), dtype=np.float32)
        tmp_p[:seq] = tmp
        d_a = _conv_matmul(lib, x_p.T, tmp_p)
        err2 = np.max(np.abs(d_a - d_a_ref))

        # Step 3: ax = x @ A → (A^T @ x^T)^T
        ax = _conv_matmul(lib, lora_a.T, x_p.T).T[:seq]
        err3 = np.max(np.abs(ax - ax_ref))

        # Step 4: d_B = ax^T @ dy
        ax_p = np.zeros((padded_seq, rank), dtype=np.float32)
        ax_p[:seq] = ax
        d_b = _conv_matmul(lib, ax_p.T, dy_p)
        err4 = np.max(np.abs(d_b - d_b_ref))

        max_err = max(err1, err2, err3, err4)
        passed = max_err < 0.01

        print(f"\n  {name}:")
        print(f"    Step 1 (dy@B^T):   err={err1:.6f}")
        print(f"    Step 2 (x^T@tmp):  err={err2:.6f}")
        print(f"    Step 3 (x@A):      err={err3:.6f}")
        print(f"    Step 4 (ax^T@dy):  err={err4:.6f}")
        print(f"    Overall: {'PASS' if passed else 'FAIL'} max_err={max_err:.6f}")
        return passed

    except RuntimeError as e:
        print(f"\n  {name}: EXCEPTION {e}")
        return False


def main():
    lib = setup_lib()
    if lib is None:
        sys.exit(1)

    print("=" * 60)
    print("ANE LoRA Gradient Pipeline Test")
    print("=" * 60)

    all_pass = True

    # q_proj / o_proj: dim=2048
    if not test_gradient(lib, 32, 2048, 8, 2048, "q/o_proj (2048→2048, rank=8, seq=32)"):
        all_pass = False

    # k_proj / v_proj: dim=256
    if not test_gradient(lib, 32, 2048, 8, 256, "k/v_proj (2048→256, rank=8, seq=32)"):
        all_pass = False

    # Longer sequence
    if not test_gradient(lib, 64, 2048, 8, 2048, "q_proj (2048→2048, rank=8, seq=64)"):
        all_pass = False

    # Different rank
    if not test_gradient(lib, 32, 2048, 16, 2048, "q_proj (2048→2048, rank=16, seq=32)"):
        all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
