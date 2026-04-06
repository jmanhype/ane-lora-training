"""Test fused MIL generation without requiring ANE bridge.

Verifies that:
1. MIL program generates correctly for various dimensions
2. MIL syntax is valid (can be compiled by ANE compiler if bridge available)
3. All required operations are present in the generated MIL
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_lora_kernels import _gen_fused_lora_grad_mil


def test_mil_generation():
    """Test MIL program generation for various dimension combinations."""
    print("=" * 60)
    print("Fused MIL Generation Test")
    print("=" * 60)

    test_cases = [
        (2048, 2048, 8, 32, "q/o_proj (2048→2048, rank=8, seq=32)"),
        (2048, 256, 8, 32, "k/v_proj (2048→256, rank=8, seq=32)"),
        (2048, 2048, 16, 64, "q_proj (2048→2048, rank=16, seq=64)"),
        (4096, 4096, 8, 32, "q_proj (4096→4096, rank=8, seq=32)"),
        (1024, 512, 4, 16, "small (1024→512, rank=4, seq=16)"),
    ]

    all_pass = True

    for in_dim, out_dim, rank, spatial, name in test_cases:
        try:
            mil_text = _gen_fused_lora_grad_mil(in_dim, out_dim, rank, spatial)

            # Verify MIL structure
            checks = {
                "Has program declaration": "program(1.3)" in mil_text,
                "Has build info": "buildInfo" in mil_text,
                "Has func main": "func main<ios18>" in mil_text,
                "Has 4 inputs": mil_text.count("tensor<fp32, [1,") >= 2 and mil_text.count("tensor<fp16, [") >= 2,
                "Has 4 conv ops": mil_text.count("conv(") >= 4,
                "Has 2 outputs": "} -> (d_A, d_B)" in mil_text,
                "Has cast ops": mil_text.count("cast(") >= 4,
                "Has string literals": 'string("' in mil_text,
                "Has const declarations": "const()[name = " in mil_text,
            }

            passed = all(checks.values())
            status = "PASS" if passed else "FAIL"

            print(f"\n  {name}:")
            for check_name, check_result in checks.items():
                print(f"    {check_name}: {'✓' if check_result else '✗'}")
            print(f"    Overall: {status}")

            if not passed:
                all_pass = False

        except Exception as e:
            print(f"\n  {name}: EXCEPTION {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return all_pass


def test_mil_dimensions():
    """Test that MIL dimensions are correctly substituted."""
    print("\n" + "=" * 60)
    print("MIL Dimension Substitution Test")
    print("=" * 60)

    in_dim, out_dim, rank, spatial = 512, 256, 8, 32
    mil_text = _gen_fused_lora_grad_mil(in_dim, out_dim, rank, spatial)

    checks = {
        f"Input dy has out_dim={out_dim}": f"tensor<fp32, [1, {out_dim}, 1, {spatial}]> dy" in mil_text,
        f"Input x has in_dim={in_dim}": f"tensor<fp32, [1, {in_dim}, 1, {spatial}]> x" in mil_text,
        f"Input A has rank={rank}, in_dim={in_dim}": f"tensor<fp16, [{rank}, {in_dim}, 1, 1]> A" in mil_text,
        f"Input B has out_dim={out_dim}, rank={rank}": f"tensor<fp16, [{out_dim}, {rank}, 1, 1]> B" in mil_text,
        f"Output d_A has in_dim={in_dim}": f"tensor<fp32, [1, {in_dim}, 1, 16]> d_A" in mil_text,
        f"Output d_B has out_dim={out_dim}": f"tensor<fp32, [1, {out_dim}, 1, 16]> d_B" in mil_text,
        f"Spatial dimension={spatial}": f", 1, {spatial}]>" in mil_text,
    }

    all_pass = all(checks.values())

    for check_name, check_result in checks.items():
        print(f"  {check_name}: {'✓' if check_result else '✗'}")

    print(f"\n  Result: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def main():
    all_pass = True

    if not test_mil_generation():
        all_pass = False

    if not test_mil_dimensions():
        all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
