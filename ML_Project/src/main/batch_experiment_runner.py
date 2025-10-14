"""
ML_Project/src/main/batch_experiment_runner.py

FIXED: Properly constructs command-line arguments from YAML config
"""

import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_file: str) -> Dict:
    """Load YAML configuration file."""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_command(base_config: Dict, exp_config: Dict, global_config: Dict) -> List[str]:
    """
    Build command-line arguments from config dictionaries.

    Args:
        base_config: Base configuration for experiment set
        exp_config: Individual experiment configuration
        global_config: Global defaults

    Returns:
        List of command-line arguments
    """
    # Merge configs: global < base < experiment
    merged = {**global_config, **base_config, **exp_config}

    # Start with Python and module
    cmd = [sys.executable, "-m", "src.main.improved_experiment_runner"]

    # Convert each config item to command-line argument
    for key, value in merged.items():
        if value is None:
            continue

        # Skip special keys that aren't CLI arguments
        if key in ['name', 'description', 'experiments', 'base_config']:
            continue

        # Convert nested configs (like lora_config, adapter_config)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value is not None:
                    arg_name = f"--{sub_key}"
                    if isinstance(sub_value, bool):
                        if sub_value:  # Only add flag if True
                            cmd.append(arg_name)
                    elif isinstance(sub_value, list):
                        cmd.append(arg_name)
                        cmd.extend([str(v) for v in sub_value])
                    else:
                        cmd.extend([arg_name, str(sub_value)])

        # Handle boolean flags
        elif isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{key}")

        # Handle lists
        elif isinstance(value, list):
            if value:  # Only add if list is not empty
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])

        # Handle regular values
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def run_experiment(exp_name: str, cmd: List[str], verbose: bool = True) -> bool:
    """
    Run a single experiment.

    Args:
        exp_name: Name of experiment
        cmd: Command to execute
        verbose: Whether to print output

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"{'='*80}")

    if verbose:
        print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print(f"✅ SUCCESS: {exp_name}")
            if verbose:
                # Print last few lines of output
                output_lines = result.stdout.strip().split('\n')
                print("\nLast output lines:")
                for line in output_lines[-10:]:
                    print(f"  {line}")
            return True
        else:
            print(f"❌ FAILED: {exp_name}")
            print(f"\nError output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {exp_name} exceeded 1 hour")
        return False
    except Exception as e:
        print(f"❌ ERROR: {exp_name}")
        print(f"  {e}")
        return False


def run_experiment_set(config: Dict, set_name: str, verbose: bool = True) -> Dict[str, bool]:
    """
    Run a complete experiment set.

    Args:
        config: Full configuration dictionary
        set_name: Name of experiment set to run
        verbose: Whether to print detailed output

    Returns:
        Dictionary mapping experiment names to success status
    """
    if set_name not in config:
        available = [k for k in config.keys() if k != 'global']
        raise ValueError(
            f"Experiment set '{set_name}' not found.\n"
            f"Available sets: {available}"
        )

    exp_set = config[set_name]
    global_config = config.get('global', {})
    base_config = exp_set.get('base_config', {})

    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT SET: {set_name}")
    print(f"# Description: {exp_set.get('description', 'N/A')}")
    print(f"{'#'*80}")

    results = {}
    experiments = exp_set.get('experiments', [])

    if not experiments:
        print("⚠️  No experiments defined in this set")
        return results

    for i, exp_config in enumerate(experiments, 1):
        exp_name = exp_config.get('name', f'experiment_{i}')

        print(f"\n[{i}/{len(experiments)}] Preparing {exp_name}...")

        # Build command
        try:
            cmd = build_command(base_config, exp_config, global_config)
        except Exception as e:
            print(f"❌ Failed to build command: {e}")
            results[exp_name] = False
            continue

        # Run experiment
        success = run_experiment(exp_name, cmd, verbose)
        results[exp_name] = success

    # Print summary
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT SET COMPLETE: {set_name}")
    print(f"{'#'*80}")

    total = len(results)
    successful = sum(1 for v in results.values() if v)
    failed = total - successful

    print(f"\nResults:")
    print(f"  Total experiments: {total}")
    print(f"  Successful: {successful} ✅")
    print(f"  Failed: {failed} ❌")

    if failed > 0:
        print(f"\nFailed experiments:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    return results


def list_experiment_sets(config: Dict):
    """List all available experiment sets."""
    print("\nAvailable Experiment Sets:")
    print("=" * 60)

    for key, value in config.items():
        if key == 'global':
            continue

        description = value.get('description', 'No description')
        num_experiments = len(value.get('experiments', []))

        print(f"\n{key}")
        print(f"  Description: {description}")
        print(f"  Number of experiments: {num_experiments}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run batch PEFT experiments from YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_configs.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--set",
        type=str,
        default="quick_test",
        help="Experiment set to run (use 'list' to see available sets)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1

    # List sets if requested
    if args.set == "list":
        list_experiment_sets(config)
        return 0

    # Run experiment set
    try:
        results = run_experiment_set(config, args.set, args.verbose)

        # Return 0 if all successful, 1 otherwise
        all_successful = all(results.values())
        return 0 if all_successful else 1

    except Exception as e:
        print(f"\n❌ Error running experiment set: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())