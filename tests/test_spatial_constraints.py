"""Test ANE spatial dimension constraints.

Discovery: ANE conv eval requires spatial >= 16 AND a multiple of 16.
This test systematically proves the constraint by sweeping dimensions.

Results on M4 Mac Mini:
  spatial < 16: ALWAYS fails (4, 8, 12)
  spatial >= 16 but not aligned: FAILS (24)
  spatial >= 16 and aligned: ALWAYS passes (16, 32, 64, 128, 256, 2048)
  Channel ratios are irrelevant — even 8→2048 works with sp=16

Requires: libane_bridge.dylib (see README for build instructions)
"""
import ctypes
import numpy as np
import sys
import os

# Allow running from repo root or tests/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_lora_kernels import _gen_conv_mil, _build_weight_blob, _pad_spatial

BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/ANE-backup/bridge/libane_bridge.dylib"))


def setup_bridge():
    """Initialize ANE bridge. Returns ctypes lib or None."""
    try:
        lib = ctypes.CDLL(BRIDGE_PATH)
    except OSError:
        print(f"Cannot load bridge from {BRIDGE_PATH}")
        print("Set ANE_BRIDGE_PATH environment variable to your libane_bridge.dylib")
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


def test_conv(lib, in_ch, out_ch, spatial):
    """Test a conv kernel. Returns (compiled, eval_ok, error)."""
    W = np.random.randn(out_ch, in_ch).astype(np.float32) * 0.01
    W_4d = np.ascontiguousarray(W.reshape(out_ch, in_ch, 1, 1), dtype=np.float32)
    blob = _build_weight_blob(W_4d)
    mil = _gen_conv_mil(in_ch, out_ch, spatial)
    mil_b = mil.encode('utf-8')
    wb = (ctypes.c_uint8 * len(blob))(*blob)
    in_sz = (ctypes.c_size_t * 1)(1 * in_ch * 1 * spatial * 4)
    out_sz = (ctypes.c_size_t * 1)(1 * out_ch * 1 * spatial * 4)
    k = lib.ane_bridge_compile(mil_b, len(mil_b), wb, len(blob), 1, in_sz, 1, out_sz)
    if not k:
        return False, False, None
    x_in = np.random.randn(1, in_ch, 1, spatial).astype(np.float32)
    lib.ane_bridge_write_input(k, 0, x_in.ctypes.data, ctypes.c_size_t(x_in.nbytes))
    ok = lib.ane_bridge_eval(k)
    err = None
    if ok:
        out = np.zeros((1, out_ch, 1, spatial), dtype=np.float32)
        lib.ane_bridge_read_output(k, 0, out.ctypes.data, ctypes.c_size_t(out.nbytes))
        expected = W @ x_in.reshape(in_ch, spatial)
        err = np.max(np.abs(out.reshape(out_ch, spatial) - expected))
    lib.ane_bridge_free(k)
    return True, ok, err


def main():
    lib = setup_bridge()
    if lib is None:
        sys.exit(1)

    np.random.seed(42)
    all_pass = True

    # ============================================================
    # Test 1: Spatial dimension sweep
    # ============================================================
    print("=" * 60)
    print("TEST 1: Spatial dimension sweep (8x8 channels)")
    print("=" * 60)
    for sp in [4, 8, 12, 16, 24, 32, 64]:
        compiled, ok, err = test_conv(lib, 8, 8, sp)
        status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
        expected = "PASS" if (sp >= 16 and sp % 16 == 0) else "FAIL"
        actual = "PASS" if ok else "FAIL"
        match = "OK" if expected == actual else "UNEXPECTED"
        print(f"  sp={sp:4d}  8->8   : {status}  [{match}]")
        if match == "UNEXPECTED":
            all_pass = False

    # ============================================================
    # Test 2: Asymmetric channels with valid spatial
    # ============================================================
    print(f"\n{'=' * 60}")
    print("TEST 2: Asymmetric channels (8<->2048) — spatial matters, not ratio")
    print("=" * 60)
    for sp in [16, 32, 64, 128, 256]:
        compiled, ok, err = test_conv(lib, 8, 2048, sp)
        status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
        print(f"  sp={sp:4d}  8->2048: {status}")
        if not ok:
            all_pass = False

        compiled, ok, err = test_conv(lib, 2048, 8, sp)
        status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
        print(f"  sp={sp:4d}  2048->8: {status}")
        if not ok:
            all_pass = False

    # ============================================================
    # Test 3: Channel ratio sweep (proving ratio doesn't matter)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("TEST 3: Channel ratio sweep (spatial=32, all should pass)")
    print("=" * 60)
    for ratio in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        compiled, ok, err = test_conv(lib, 8, 8 * ratio, 32)
        status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
        print(f"  8->{8*ratio:5d} (ratio {ratio:4d}x): {status}")
        if not ok:
            all_pass = False

    # ============================================================
    # Test 4: _pad_spatial correctness
    # ============================================================
    print(f"\n{'=' * 60}")
    print("TEST 4: _pad_spatial() function correctness")
    print("=" * 60)
    test_cases = [
        (1, 16), (4, 16), (8, 16), (12, 16), (15, 16), (16, 16),
        (17, 32), (24, 32), (31, 32), (32, 32), (48, 48), (64, 64),
        (100, 112), (256, 256), (512, 512),
    ]
    for inp, expected in test_cases:
        actual = _pad_spatial(inp)
        status = "PASS" if actual == expected else f"FAIL (got {actual})"
        print(f"  _pad_spatial({inp:4d}) = {actual:4d}  {status}")
        if actual != expected:
            all_pass = False

    # ============================================================
    # Test 5: Actual LoRA gradient shapes
    # ============================================================
    print(f"\n{'=' * 60}")
    print("TEST 5: LoRA gradient shapes (rank=8, dim=2048, seq=32)")
    print("=" * 60)
    seq, rank, dim = 32, 8, 2048

    shapes = [
        (dim, rank, seq, "Step 1: B[rank,out] @ dy^T[out,seq]"),
        (seq, dim, _pad_spatial(rank), "Step 2: x^T[in,seq] @ tmp[seq,rank] (padded)"),
        (dim, rank, seq, "Step 3: A^T[rank,in] @ x^T[in,seq]"),
        (seq, rank, dim, "Step 4: ax^T[rank,seq] @ dy[seq,out]"),
    ]
    for in_ch, out_ch, sp, desc in shapes:
        compiled, ok, err = test_conv(lib, in_ch, out_ch, sp)
        status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
        print(f"  {desc}: {status}")
        if not ok:
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
