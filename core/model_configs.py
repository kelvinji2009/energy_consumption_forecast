"""
模型配置文件 - 为不同算法提供最佳默认参数
"""

# 不同模型的最佳默认参数配置
MODEL_CONFIGS = {
    "LightGBM": {
        "input_chunk_length": 24 * 3,  # 3天历史数据，LightGBM不需要太长序列
        "output_chunk_length": 24,     # 预测1天
        "model_params": {
            "random_state": 42,
            "force_reset": True,
            "lags_future_covariates": [0, 23],  # 当前和23小时后的协变量
        }
    },
    
    "TiDE": {
        "input_chunk_length": 24 * 5,  # 5天历史数据，适合捕捉周期性
        "output_chunk_length": 24,     # 预测1天
        "model_params": {
            "hidden_size": 128,         # 增大隐藏层，提升表达能力
            "random_state": 42,
            "force_reset": True,
        }
    },
    
    "LSTM": {
        "input_chunk_length": 24 * 7,  # 7天历史数据，LSTM需要更长序列学习模式
        "output_chunk_length": 24,     # 预测1天
        "model_params": {
            "model": "LSTM",
            "training_length": 24 * 7,  # 训练序列长度
            "random_state": 42,
            "force_reset": True,
        }
    },
    
    "TFT": {
        "input_chunk_length": 24 * 7,  # 7天历史数据，TFT需要长序列学习复杂依赖
        "output_chunk_length": 24,     # 预测1天
        "model_params": {
            "hidden_size": 128,         # 增大隐藏层
            "lstm_layers": 2,           # 增加LSTM层数
            "num_attention_heads": 8,   # 增加注意力头数
            "dropout": 0.1,
            "batch_size": 32,           # 增大批次大小
            "random_state": 42,
            "force_reset": True,
        }
    },
    
    "TFT (No Past Covariates)": {
        "input_chunk_length": 24 * 7,  # 7天历史数据
        "output_chunk_length": 24,     # 预测1天
        "model_params": {
            "hidden_size": 96,          # 无过去协变量时稍小的隐藏层
            "lstm_layers": 2,
            "num_attention_heads": 6,
            "dropout": 0.1,
            "batch_size": 32,
            "random_state": 42,
            "force_reset": True,
        }
    }
}

def get_model_config(model_type: str) -> dict:
    """
    获取指定模型类型的配置
    
    Args:
        model_type: 模型类型名称
        
    Returns:
        包含模型配置的字典
        
    Raises:
        ValueError: 如果模型类型不支持
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_type].copy()

def get_supported_models() -> list:
    """获取所有支持的模型类型列表"""
    return list(MODEL_CONFIGS.keys())

def update_model_config(model_type: str, **kwargs) -> dict:
    """
    更新模型配置参数
    
    Args:
        model_type: 模型类型
        **kwargs: 要更新的参数
        
    Returns:
        更新后的配置字典
    """
    config = get_model_config(model_type)
    
    # 更新顶级参数
    for key in ['input_chunk_length', 'output_chunk_length']:
        if key in kwargs:
            config[key] = kwargs[key]
    
    # 更新模型参数
    if 'model_params' in kwargs:
        config['model_params'].update(kwargs['model_params'])
    
    return config