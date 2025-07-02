# 工业能耗预测与异常检测

本项目提供了一个完整的时间序列分析流程，包括模拟数据生成、模型训练、能耗预测和异常检测。项目使用 `darts` 库来展示如何应用不同的模型
（如 Temporal Fusion Transformer (TFT) 和 LightGBM）来解决真实的工业问题。

## 📋 功能特性

- **模拟数据生成**: 创建一个逼真的数据集 (`simulated_plant_data.csv`)，模拟包含产量、温度和湿度的每小时工业数据，其中涵盖了日、周和季节性周期，并注入了异常点。
- **时间序列预测**: 训练和评估多种先进的预测模型（TFT, LightGBM），以预测未来的能源消耗。
- **协变量支持**: 演示了如何使用过去、现在和未来的协变量（如产量水平、温度和基于时间的特征）来提高模型的准确性。
- **异常检测**: 实现了一个基于残差的异常检测系统。模型会预测预期的能源使用量，而显著的偏差则被标记为异常。
- **可视化**: 生成图表以可视化历史数据、检测到的异常和未来的预测。

## 📂 项目结构

```
/
├── data/
│   └── simulated_plant_data.csv    # 原始模拟数据
├── demo/
│   ├── 01_data_preprocessing.py    # 清洗数据和特征工程
│   ├── 02_train_and_evaluate*.py   # 用于训练不同模型的脚本
│   ├── 03_anomaly_detection*.py    # 用于异常检测和预测的脚本
│   ├── processed_data.csv          # 用于建模的已处理数据
│   ├── models/                     # 存储训练好的模型文件
│   └── plots/                      # 存储输出的可视化图表
├── .gitignore
├── generate_data.py                # 用于生成初始数据集的脚本
├── LICENSE
└── README.md
```

## 🚀 快速开始

### 环境准备（conda）

请确保您已安装 conda，并已经创建了对应darts env。

```bash
conda env list | grep darts
```
如果未找到darts env，使用以下命令创建
```bash
conda create -n darts python=3.9 -y
```

本项目依赖以下库，您可以使用 pip 进行安装：

```bash
conda run -n darts pip install pandas numpy "darts[torch]" matplotlib scikit-learn joblib
```
*注意: `darts[torch]` 会确保 PyTorch (TFT 模型的一个依赖项) 被正确安装。*

### 分步工作流

1.  **生成数据**:
    首先，在项目的根目录下运行生成脚本来创建模拟数据集。

    ```bash
    conda run -n darts python generate_data.py
    ```
    这将在 `data/` 目录下创建 `simulated_plant_data.csv` 文件。

2.  **预处理数据**:
    接下来，运行预处理脚本。此脚本应在 `demo` 目录下运行。

    ```bash
    cd demo
    conda run -n darts python 01_data_preprocessing.py
    ```
    这将在 `demo/` 目录下创建 `processed_data.csv` 文件。

3.  **训练预测模型**:
    您可以选择训练多个模型中的一个。例如，要训练 LightGBM 模型，请在 `demo` 目录下运行以下命令：

    ```bash
    # 训练 LightGBM 模型
    conda run -n darts python 02_train_and_evaluate_lgbm.py

    # 或者，训练 Temporal Fusion Transformer (TFT) 模型
    conda run -n darts python 02_train_and_evaluate.py
    ```
    训练好的模型将保存在 `demo/models/` 目录下。

4.  **检测异常并预测未来消耗**:
    训练模型后，运行相应的异常检测脚本。例如，如果您在上一步中训练了 LightGBM 模型：

    ```bash
    conda run -n darts python 03_anomaly_detection_lgbm.py
    ```
    此脚本将使用训练好的模型在历史数据中查找异常，并生成2025年上半年的能耗预测。显示结果的最终图表将保存在 `demo/plots/` 目录下。

## 🛠️ 方法论

### 预测

该项目利用 `darts` 库进行时间序列预测。它将能耗作为目标变量，并使用其他数据点作为协变量：
- **过去协变量 (Past Covariates)**: 截至当前已知的历史数据（例如 `production_units`, `temperature_celsius`）。
- **未来协变量 (Future Covariates)**: 事先已知的数据（例如，一天中的小时，一周中的天）。

通过向模型提供这些协变量，它们可以学习更复杂的关系并产生更准确的预测。

### 异常检测

异常检测方法基于预测残差。工作流程如下：
1.  训练好的模型为训练集或验证集生成历史预测。
2.  计算每个时间步的实际值与模型预测之间的绝对差值（残差）。
3.  在这些残差上拟合一个 `QuantileDetector`。它确定一个阈值（例如，第98个百分位数），任何高于此阈值的残差都被视为异常。
4.  这个拟合好的检测器随后用于对新的或未见过的数据进行评分和标记异常。

这种方法是有效的，因为它将“异常”定义为模型在学习了系统的正常模式后无法预测的事件。

## 📄 许可证

本项目根据 LICENSE 文件中的条款进行许可。
