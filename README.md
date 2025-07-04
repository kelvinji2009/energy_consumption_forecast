# 工业能耗预测与异常检测

本项目提供了一个完整的时间序列分析流程，包括模拟数据生成、模型训练、能耗预测和异常检测。项目使用 `darts` 库来展示如何应用不同的模型（如 **Temporal Fusion Transformer (TFT)**、**LSTM**、**TiDE** 和 **LightGBM**）来解决真实的工业问题。

## 📋 功能特性

- **模拟数据生成**: 创建一个逼真的数据集 (`simulated_plant_data.csv`)，模拟包含产量、温度和湿度的每小时工业数据，其中涵盖了日、周和季节性周期，并注入了异常点。
- **多模型时间序列预测**: 训练和评估多种先进的预测模型（TFT, LSTM, TiDE, LightGBM），以预测未来的能源消耗。
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
│   ├── 02_train_and_evaluate*.py   # 用于训练不同模型的脚本 (TFT, LSTM, TiDE, LightGBM)
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
*注意: `darts[torch]` 会确保 PyTorch (TFT 和 LSTM 模型的一个依赖项) 被正确安装。*

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
    您可以选择训练四个模型中的任何一个。在 `demo` 目录下运行以下命令：

    ```bash
    # 训练 Temporal Fusion Transformer (TFT) 模型
    conda run -n darts python 02_train_and_evaluate.py

    # 训练 LightGBM 模型
    conda run -n darts python 02_train_and_evaluate_lgbm.py

    # 训练 LSTM 模型
    conda run -n darts python 02_train_and_evaluate_lstm.py
    
    # 训练 TiDE 模型
    conda run -n darts python 02_train_and_evaluate_tide.py
    ```
    训练好的模型将保存在 `demo/models/` 目录下。

4.  **检测异常并预测未来消耗**:
    训练模型后，运行相应的异常检测脚本。例如，如果您训练了 **LSTM** 模型：

    ```bash
    # 使用 LSTM 模型进行异常检测
    conda run -n darts python 03_anomaly_detection_lstm.py

    # 或者，如果您训练了 TiDE 模型
    conda run -n darts python 03_anomaly_detection_tide.py
    ```
    此脚本将使用训练好的模型在历史数据中查找异常，并生成2025年上半年的能耗预测。显示结果的最终图表将保存在 `demo/plots/` 目录下。

## 🛠️ 方法论

### 预测模型

该项目利用 `darts` 库进行时间序列预测，并实现了以下四个模型：

- **Temporal Fusion Transformer (TFT)**: 一个基于注意力机制的深度学习模型，专为多水平时间序列预测设计。它能够捕捉复杂的长期依赖关系，并整合静态元数据和多种协变量。
- **LSTM (Long Short-Term Memory)**: 一种循环神经网络（RNN），非常适合处理和预测时间序列数据中的序列依赖性。
- **TiDE (Time-series Dense Encoder)**: 一个新颖的、完全基于密集连接的模型，它在保持高性能的同时，比基于注意力的模型更高效。
- **LightGBM**: 一个基于树的学习算法，通常用于表格数据，但通过特征工程（如滞后特征），它也能非常有效地用于时间序列预测。

所有模型都将能耗作为目标变量，并使用其他数据点作为协变量：
- **过去协变量 (Past Covariates)**: 截至当前已知的历史数据（例如 `production_units`, `temperature_celsius`）。
- **未来协变量 (Future Covariates)**: 事先已知的数据（例如，一天中的小时，一周中的天）。

### 异常检测

异常检测方法基于预测残差。工作流程如下：
1.  训练好的模型为训练集或验证集生成历史预测。
2.  计算每个时间步的实际值与模型预测之间的绝对差值（残差）。
3.  在这些残差上拟合一个 `QuantileDetector`。它确定一个阈值（例如，第98个百分位数），任何高于此阈值的残差都被视为异常。
4.  这个拟合好的检测器随后用于对新的或未见过的数据进行评分和标记异常。

这种方法是有效的，因为它将“异常”定义为模型在学习了系统的正常模式后无法预测的事件。

## 📄 许可证

本项目根据 LICENSE 文件中的条款进行许可。

---

## 🚀 系统运行指南

本项目已扩展为一个完整的能耗预测与异常检测系统，包含API服务、后台管理和异步训练。以下是启动和运行整个系统的步骤：

**前提条件：**

*   已安装 `conda`。
*   已创建并激活 `darts` conda 环境，并安装了所有项目依赖。
*   已安装并运行 `PostgreSQL` 数据库。
*   已安装并运行 `Redis` 服务器（作为 Celery 的消息代理和结果后端）。

**启动步骤：**

1.  **设置数据库环境变量**：
    在您将要运行后端 API 和 Celery Worker 的终端中，设置 `DATABASE_URL` 环境变量。
    ```bash
    export DATABASE_URL="postgresql://kelvinji:mitnick888@localhost:5432/energy_forecast_db"
    ```
    请根据您的实际 PostgreSQL 用户名、密码和数据库名进行修改。

2.  **初始化数据库表和数据** (仅首次运行或需要重置时执行)：
    从项目根目录执行：
    ```bash
    conda run -n darts python -m database.database
    conda run -n darts python -m database.insert_initial_data
    ```
    这将创建所需的数据库表并插入一个默认资产和模型记录。

3.  **启动 Celery Worker** (在**一个单独的终端**中，从项目根目录执行)：
    ```bash
    conda run -n darts celery -A celery_worker.celery_app worker -l info
    ```
    这将启动 Celery 任务处理器，它会监听 Redis 队列并执行模型训练任务。

4.  **启动后端 API 服务** (在**另一个单独的终端**中，从项目根目录执行)：
    ```bash
    ./api_server/start_server.sh
    ```
    这将启动 FastAPI 后端服务，包括预测 API 和管理 API。

5.  **启动前端开发服务器** (在**第三个单独的终端**中，进入 `admin_frontend` 目录后执行)：
    ```bash
    cd admin_frontend
    npm run dev
    ```
    这将启动 React 前端应用。

**访问系统：**

*   **后端 API 文档 (Swagger UI)**：在浏览器中访问 `http://127.0.0.1:8000/docs`
*   **后台管理系统 (前端 UI)**：在浏览器中访问 `http://localhost:5173/` (如果 `npm run dev` 输出的端口不同，请以实际端口为准)

**测试预测 API (需要 API Key)：**

1.  通过后台管理系统前端 UI (API 密钥页面) 生成一个 API 密钥。
2.  使用 `api_client_test.py` 脚本进行测试，确保在请求头中包含 `Authorization: Bearer <YOUR_API_KEY>`。
    ```bash
    conda run -n darts python api_client_test.py
    ```