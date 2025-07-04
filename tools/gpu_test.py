import torch
import darts

# 1. 检查 PyTorch 版本 (应该是一个新版本，如 2.x)
print(f"PyTorch version: {torch.__version__}")

# 2. 检查 Darts 版本 (应该是一个新版本，如 0.2x.x)
print(f"Darts version: {darts.__version__}")

# 3. 检查 MPS (Apple Silicon GPU) 是否可用 (这步最关键！)
if torch.backends.mps.is_available():
    print("✅ Congratulations! Your M3 GPU (MPS backend) is available to PyTorch.")
    device = torch.device("mps")
else:
    print("❌ MPS not available. You'll be using the CPU.")
    device = torch.device("cpu")

print(f"Darts will use device: {device}")
