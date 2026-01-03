import os
import numpy as np

# 根目录（相对 or 绝对路径都可以）
ROOT = "dataset/QuickDraw414k/coordinate_files/train"

# 指定一个具体的 npy 文件
path = os.path.join(ROOT, "airplane", "airplane_0.npy")

# 1. 先检查文件是否存在
if not os.path.exists(path):
    raise FileNotFoundError(f"文件不存在: {path}")

# 2. 加载 npy（QuickDraw 通常需要 allow_pickle=True）
data = np.load(path, allow_pickle=True)

# 3. 基本信息
print("=== 基本信息 ===")
print("type:", type(data))
print("dtype:", data.dtype)
print("shape:", getattr(data, "shape", None))
print("len:", len(data))

# 4. 查看前几个元素（避免终端被刷爆）
print("\n=== 前 100 个元素 ===")
for i in range(min(100, len(data))):
    print(f"sample[{i}]:")
    print(data[i])
    print("-" * 40)

# 5. 如果是单个样本，查看更细节
print("\n=== 第 0 个样本细节 ===")
sample = data[0]
print("sample type:", type(sample))
print("sample shape:", getattr(sample, "shape", None))
print(sample)
