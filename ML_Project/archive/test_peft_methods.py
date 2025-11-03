"""
Quick test script to verify all PEFT methods are working correctly.
"""

import subprocess
import sys


def test_peft_method(method: str, additional_args: list = None):
    """Test a single PEFT method with minimal configuration."""
    print(f"\n{'=' * 60}")
    print(f"Testing {method.upper()} method...")
    print('=' * 60)

    cmd = [
        sys.executable, "-m", "src.main.main_experiment_runner",
        "--dataset", "sst2",
        "--model_name", "distilbert-base-uncased",
        "--peft_method", method,
        "--epochs", "1",  # Just 1 epoch for testing
        "--batch_size", "32"  # Larger batch for faster testing
    ]

    if additional_args:
        cmd.extend(additional_args)

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {method} test PASSED")
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            print("Last output lines:")
            for line in output_lines[-5:]:
                print(f"  {line}")
        else:
            print(f"✗ {method} test FAILED")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ {method} test FAILED with exception: {e}")
        return False


def main():
    """Run tests for all PEFT methods."""
    print("PEFT Methods Test Suite")
    print("=" * 80)

    tests = [
        ("lora", ["--lora_r", "4", "--lora_alpha", "8"]),
        ("adapter", ["--adapter_reduction_factor", "16"]),
        ("hybrid", ["--lora_r", "4", "--adapter_reduction_factor", "16"])
    ]

    results = {}

    for method, args in tests:
        success = test_peft_method(method, args)
        results[method] = success

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for method, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{method.upper()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests PASSED! Your PEFT implementation is ready.")
        print("\nNext steps:")
        print("1. Run full experiments: python run_all_experiments.py")
        print("2. Analyze results: python analyze_results.py")
    else:
        print("\n❌ Some tests FAILED. Please check the error messages above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())