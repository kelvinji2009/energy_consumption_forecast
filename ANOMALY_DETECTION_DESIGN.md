# 异常检测 API 完整实现设计方案

本文档详细规划了在现有能耗预测与异常检测系统中，实现异常检测 API 的具体方案。

## 1. 核心思路

异常检测将基于预测残差。其基本流程如下：

1.  **预测模型训练**：利用历史数据训练一个能耗预测模型（已实现）。
2.  **残差计算**：使用训练好的预测模型在历史数据上生成预测，并计算实际值与预测值之间的绝对差值（残差）。
3.  **检测器拟合**：在这些历史残差上拟合一个异常检测器（例如 Darts 的 `QuantileDetector`），以学习“正常”残差的分布并确定异常阈值。
4.  **实时异常识别**：在接收到实时数据流时，首先使用预测模型进行预测，计算残差，然后用已拟合的检测器来识别这些残差中的异常点。

## 2. 核心挑战：检测器的存储与加载

为了实现异常检测 API，我们需要一种机制来存储和加载与每个预测模型关联的**已拟合的 `QuantileDetector` 对象**。

## 3. 设计方案：存储与加载已拟合的检测器对象

### 3.1 在模型训练时保存检测器

*   **位置**：`celery_worker/tasks.py`
*   **逻辑**：
    *   在 `train_model_for_asset` Celery 任务中，当预测模型训练完成后，利用训练数据（或专门的验证数据）生成历史预测。
    *   计算这些历史预测的残差。
    *   实例化并 `fit()` 一个 `QuantileDetector` 到这些残差上（例如，使用 `high_quantile=0.98`）。
    *   使用 `joblib` 将这个**已拟合的 `QuantileDetector` 对象**保存到一个单独的文件中。建议将其保存在与预测模型相同的目录下，命名约定为 `detector.joblib`。
    *   **数据库影响**：无需修改 `models_table` 结构。通过约定，每个模型记录的 `path` 字段指向的目录下，如果存在 `detector.joblib`，则表示该模型支持异常检测。

### 3.2 在 API 服务启动时加载检测器

*   **位置**：`api_server/main.py` 的 `lifespan` 函数。
*   **逻辑**：
    *   当加载预测模型时，同时尝试加载其对应目录下的 `detector.joblib` 文件。
    *   将加载的预测模型和检测器都存储在 `app.state.model_cache` 中。建议的数据结构为：
        ```python
        app.state.model_cache[asset_id] = {
            'model': model_obj,
            'detector': detector_obj # 如果存在则加载，否则为 None
        }
        ```

### 3.3 实现 `/assets/{asset_id}/detect_anomalies` 端点

*   **位置**：`api_server/main.py`
*   **输入**��接收一段实时或历史数据流 (`data_stream`)，其结构与预测 API 的输入类似。
*   **逻辑**：
    a.  从 `app.state.model_cache` 中获取指定 `asset_id` 的**预测模型**和**已拟合的检测器**。
    b.  **数据转换**：将输入的 `data_stream` 转换为 Darts `TimeSeries` 对象。
    c.  **预测**：使用预测模型对 `data_stream` 进行历史预测（即预测每个时间点在模型看来“应该”是什么值）。
    d.  **残差计算**：计算实际值与预测值之间的绝对差值（残差）。
    e.  **异常识别**：使用已加载的检测器 (`detector.detect(residuals)`) 来识别这些残差中的异常点。
    f.  **结果筛选**：根据异常标志，从原始 `data_stream` 中筛选出被标记为异常的数据点。
    g.  **格式化输出**：格式化并返回异常列表，可以包含异常发生的时间、实际值以及简单的异常原因（例如“高于预期”或“低于预期”）。
*   **输出**：返回一个异常数据点的列表。

## 4. 技术细节与考虑

*   **`QuantileDetector` 的 `high_quantile`**：在 `celery_worker/tasks.py` 中，可以硬编码这个值（例如 `0.98`），或者未来将其作为训练任务的参数，使其可配置。
*   **异常原因**：在返回异常时，可以根据残差是正还是负来判断是“高���预期”还是“低于预期”，提供更具描述性的 `reason` 字段。
*   **性能**：异常检测 API 的性能要求与预测 API 类似，模型和检测器的预加载和缓存至关重要。
*   **协变量处理**：在进行历史预测以计算残差时，需要确保正确地生成和提供 `past_covariates` 和 `future_covariates`，这与预测 API 的逻辑类似。

## 5. 实施路线图（异常检测 API）

1.  **修改 `celery_worker/tasks.py`**：
    *   在模型训练和评估之后，添加生成历史残差、拟合 `QuantileDetector` 并保存其对象到 `detector.joblib` 的逻辑。
2.  **修改 `api_server/main.py`**：
    *   在 `lifespan` 函数中，当加载预测模型时，同时加载对应的 `detector.joblib` 文件，并将其存储在 `app.state.model_cache` 中。
    *   实现 `POST /assets/{asset_id}/detect_anomalies` 端点，包含上述的异常检测逻辑。
3.  **更新 `api_client_test.py`** (可选)：
    添加一个测试用例，调用新的异常检测 API。
4.  **更新前端 UI** (可选)：
    在后台管理系统中添加一个页面或功能，用于展示异常检测结果或触发异常检测。
