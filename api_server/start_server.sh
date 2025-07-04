#!/bin/bash

# 这是一个用于启动API服务的脚本。
# 它解决了`conda run`无法正确设置所有环境变量的问题。

# 获取conda的安装路径，这使得脚本更具可移植性
CONDA_BASE=$(conda info --base)

# 激活conda环境
echo "Activating conda environment: darts"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate darts

# 检查激活是否成功
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Exiting."
    exit 1
fi

# 切换到脚本所在的目录，以确保相对路径正确
cd "$(dirname "$0")"

# 启动Uvicorn服务
echo "Starting Uvicorn server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
