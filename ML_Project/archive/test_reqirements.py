# Test script: verify_install.py
import torch
import transformers
import adapters
import peft
import datasets

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ Adapters: {adapters.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ Datasets: {datasets.__version__}")
print("\nAll libraries installed successfully!")