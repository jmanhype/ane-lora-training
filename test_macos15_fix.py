#!/usr/bin/env python3
"""
Test script to verify macOS 15 ANE bridge fix.

This script tests the ios17 MIL program target fix for macOS 15 compatibility.
It runs a simple conv operation and checks if it succeeds.

Usage:
    python test_macos15_fix.py

Expected output:
    - PASSED: ANE eval succeeds with ios17 target
    - FAILED: ANE eval fails with detailed error information
"""

import ctypes
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ane_lora_kernels import _conv_matmul, _gen_conv_mil

BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/libane_bridge.dylib"))


def setup_lib():
    """Initialize ANE bridge for testing."""
    try:
        lib = ctypes.CDLL(BRIDGE_PATH)
    except OSError as e:
        print(f"❌ Cannot load bridge from {BRIDGE_PATH}")
        print(f"   Error: {e}")
        print(f"   Set ANE_BRIDGE_PATH environment variable to correct path")
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
        print(f"❌ Bridge init failed: {rc}")
        return None
    return lib


def test_ios17_target():
    """Test that MIL programs use ios17 target."""
    print("\n" + "="*60)
    print("Testing MIL Program Version Target")
    print("="*60)

    # Generate a simple conv MIL program
    mil_text = _gen_conv_mil(16, 16, 32)

    # Check if it uses ios17
    if "func main<ios17>" in mil_text:
        print("✅ MIL program uses ios17 target (CORRECT for macOS 15)")
        return True
    elif "func main<ios18>" in mil_text:
        print("⚠️  MIL program uses ios18 target (may fail on macOS 15)")
        print("   The fix has not been applied correctly")
        return False
    else:
        print("❌ MIL program uses unknown target")
        print(f"   Found: {mil_text[:200]}...")
        return False


def test_simple_conv():
    """Test a simple conv operation that should work on macOS 15."""
    print("\n" + "="*60)
    print("Testing Simple Conv Operation (16x16, spatial=32)")
    print("="*60)

    lib = setup_lib()
    if lib is None:
        return False

    np.random.seed(42)
    W = np.random.randn(16, 16).astype(np.float32) * 0.1
    x = np.random.randn(16, 32).astype(np.float32)

    try:
        print("  Compiling MIL program...")
        result = _conv_matmul(lib, W, x)
        expected = W @ x
        err = np.max(np.abs(result - expected))

        if err < 0.01:
            print(f"✅ PASSED: Conv operation succeeded")
            print(f"   Max error: {err:.6f}")
            print(f"   Result shape: {result.shape}")
            return True
        else:
            print(f"❌ FAILED: Conv operation succeeded but numerical error too high")
            print(f"   Max error: {err:.6f} (expected < 0.01)")
            return False

    except RuntimeError as e:
        print(f"❌ FAILED: Conv operation failed with error:")
        print(f"   {e}")
        return False

    finally:
        # Clean up
        if lib:
            lib.ane_bridge_free.argtypes = [ctypes.c_void_p]


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("macOS 15 ANE Bridge Fix Verification")
    print("="*60)
    print(f"Bridge path: {BRIDGE_PATH}")

    # Check macOS version
    import platform
    macos_version = platform.mac_ver()[0]
    print(f"macOS version: {macos_version}")

    if not macos_version.startswith("15."):
        print(f"⚠️  Warning: This fix is for macOS 15.x")
        print(f"   Current version: {macos_version}")

    # Test 1: Check MIL program target
    test1_passed = test_ios17_target()

    # Test 2: Test simple conv operation
    test2_passed = test_simple_conv()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"MIL Program Target: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Simple Conv Test:   {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! macOS 15 fix is working correctly.")
        return 0
    else:
        print("\n❌ Some tests FAILED. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
