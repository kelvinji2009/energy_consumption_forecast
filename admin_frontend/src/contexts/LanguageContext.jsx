import React, { createContext, useContext, useState, useEffect } from 'react';

const LanguageContext = createContext();

const translations = {
  zh: {
    // Navigation
    nav: {
      home: '首页',
      forecast: '能耗预测',
      anomaly: '异常检测',
      training: '模型训练',
      models: '模型管理',
      assets: '资产管理',
      apiKeys: 'API 密钥',
      apiDocs: 'API 文档'
    },
    
    // Home page
    home: {
      title: '能耗预测与异常检测管理系统',
      subtitle: '基于机器学习的工业能耗智能分析平台',
      features: {
        prediction: {
          title: '智能预测',
          desc: '支持多种算法的能耗预测模型'
        },
        anomaly: {
          title: '异常检测',
          desc: '实时识别能耗异常模式'
        },
        async: {
          title: '异步处理',
          desc: '高效的后台任务处理'
        },
        management: {
          title: '资产管理',
          desc: '统一的设备和资产管理'
        },
        visualization: {
          title: '数据可视化',
          desc: '直观的图表和报告'
        },
        security: {
          title: '安全认证',
          desc: 'API密钥和权限管理'
        }
      },
      getStarted: '开始使用',
      learnMore: '了解更多'
    },

    // Training page
    training: {
      title: '模型训练',
      selectAsset: '选择资产',
      chooseAsset: '请选择资产',
      selectModelType: '选择模型类型',
      selectAlgorithm: '选择算法',
      chooseAlgorithm: '请选择算法',
      uploadTrainingData: '上传训练数据',
      algorithms: {
        'LightGBM': {
          label: 'LightGBM',
          desc: '轻量级梯度提升机，适合快速训练和高精度预测'
        },
        'TiDE': {
          label: 'TiDE',
          desc: '时间序列密集编码器，专为长期预测设计'
        },
        'LSTM': {
          label: 'LSTM',
          desc: '长短期记忆网络，擅长处理序列数据'
        },
        'TFT': {
          label: 'TFT',
          desc: '时间融合变换器，支持多变量时间序列预测'
        },
        'TFT (No Past Covariates)': {
          label: 'TFT (无历史协变量)',
          desc: '简化版TFT，不使用历史协变量'
        }
      },
      dataInputMethod: '数据输入方式',
      uploadCsv: '上传 CSV 文件',
      uploadCSV: '上传 CSV 文件',
      s3Path: 'S3 路径',
      trainingDataFile: '训练数据文件',
      chooseFile: '选择文件',
      noFileSelected: '未选择任何文件',
      fileSelected: '已选择文件',
      s3DataPath: 'S3数据路径',
      s3Placeholder: '例如: s3://bucket/path/to/data.csv',
      epochs: '训练轮数',
      epochsHint: '建议值：LightGBM 10-50轮，深度学习模型 20-100轮',
      startTraining: '开始训练',
      starting: '正在启动...',
      taskRunning: '任务运行中...',
      noAssets: '没有可用资产',
      errors: {
        failedToLoadAssets: '加载资产失败',
        noFile: '请选择文件',
        noS3Path: '请输入S3路径',
        failedToStart: '启动训练失败'
      },
      status: {
        PENDING: '等待中',
        PROGRESS: '训练中',
        SUCCESS: '训练完成',
        FAILURE: '训练失败',
        RETRY: '重试中',
        REVOKED: '已取消'
      },
      trainingCompleted: '训练完成！',
      trainingFailed: '训练失败',
      trainingInProgress: '训练进行中',
      statusLabel: '状态',
      parameters: {
        inputSequenceLength: '输入序列长度',
        outputSequenceLength: '输出序列长度',
        inputSequenceLengthHint: '用于训练的历史数据长度',
        outputSequenceLengthHint: '预测的未来数据长度',
        title: '模型参数配置',
        // 基础参数
        input_chunk_length: '输入序列长度',
        output_chunk_length: '输出序列长度',
        input_chunk_length_desc: '用于预测的历史数据长度（小时）',
        output_chunk_length_desc: '预测的未来时间长度（小时）',
        // 模型特定参数
        random_state: '随机种子',
        random_state_desc: '确保结果可重现',
        hidden_size: '隐藏层大小',
        hidden_size_desc: '神经网络隐藏层的神经元数量',
        training_length: '训练长度',
        training_length_desc: '训练时使用的序列长度',
        lstm_layers: 'LSTM层数',
        lstm_layers_desc: 'LSTM层的数量',
        num_attention_heads: '注意力头数',
        num_attention_heads_desc: '多头注意力机制的头数',
        dropout: 'Dropout率',
        dropout_desc: '防止过拟合的dropout比例',
        batch_size: '批次大小',
        batch_size_desc: '每个训练批次的样本数量',
        // 参数分类
        category: {
          model: '模型参数',
          training: '训练参数',
          data: '数据参数'
        }
      }
    },

    // Forecast page
    forecast: {
      title: '能耗预测',
      selectAsset: '选择资产',
      chooseAsset: '请选择资产',
      selectModel: '选择模型',
      chooseModel: '请选择模型',
      forecastHours: '预测步长（小时）',
      trained: '训练时间',
      dataInputMethod: '数据输入方式',
      uploadCSV: '上传 CSV 文件',
      s3Path: 'S3 路径',
      uploadHistoricalData: '上传历史数据 CSV',
      s3PathInput: 'S3 数据路径',
      startForecast: '开始预测',
      forecasting: '预测中...',
      results: '预测结果',
      historicalEnergy: '历史能耗',
      predictedEnergy: '预测能耗',
      timestamp: '时间戳',
      energyKwh: '能耗 (kWh)',
      chartTitle: '能耗预测图表',
      fileSelected: '已选择',
      lastSelectedFile: '上次选择的文件',
      reselect: '请重新选择',
      waitingResults: '等待预测结果...',
      configureAndPredict: '配置参数并点击预测后，结果将在此显示',
      visualAnalysis: '支持历史数据和预测数据的可视化分析'
    },

    // Anomaly Detection page
    anomaly: {
      title: '异常检测',
      selectAsset: '选择资产',
      selectModel: '选择模型',
      trained: '训练时间',
      sensitivity: '异常检测敏感度',
      moreAnomalies: '更多异常',
      fewerAnomalies: '更少异常',
      sensitivityDesc: '调整异常检测的严格程度，数值越高检测越严格',
      dataInputMethod: '数据输入方式',
      uploadCSV: '上传 CSV 文件',
      s3Path: 'S3 路径',
      uploadHistoricalData: '上传历史数据 CSV',
      s3PathInput: 'S3 数据路径',
      startDetection: '开始检测',
      detecting: '检测中...',
      noDetectors: '该资产没有可用的异常检测器',
      noModels: '没有可用的模型',
      historicalEnergy: '历史能耗',
      anomalies: '异常点',
      timestamp: '时间戳',
      energyKwh: '能耗 (kWh)',
      chartTitle: '异常检测结果'
    },

    // Models page
    models: {
      title: '模型管理',
      loading: '加载模型中...',
      refreshing: '刷新中...',
      refresh: '刷新列表',
      loadError: '加载模型失败',
      noModels: '暂无模型',
      startTraining: '您可以在"模型训练"页面开始新的训练任务',
      modelId: '模型 ID',
      asset: '资产',
      type: '类型',
      version: '版本',
      created: '创建时间',
      mape: 'MAPE 误差',
      s3Path: 'S3 路径',
      status: {
        completed: '已完成',
        training: '训练中',
        pending: '等待中',
        failed: '失败'
      },
      trainingParameters: '训练参数配置',
      parameterType: '类型'
    },

    // Assets page
    assets: {
      title: '资产管理',
      loading: '加载资产中...',
      fetchError: '获取资产失败',
      saveError: '保存资产失败',
      deleteError: '删除资产失败',
      createNew: '创建新资产',
      editAsset: '编辑资产',
      createAsset: '创建资产',
      id: 'ID',
      name: '名称',
      description: '描述',
      modelCount: '模型数量',
      actions: '操作',
      assetId: '资产 ID',
      assetName: '资产名称',
      assetDescription: '资产描述',
      idCannotChange: 'ID 创建后无法修改',
      cancel: '取消',
      create: '创建',
      saveChanges: '保存更改',
      deleteConfirm: '确定要删除资产 {id} 吗？此操作无法撤销。'
    },

    // API Keys page
    apiKeys: {
      title: 'API 密钥管理',
      loading: '加载密钥中...',
      loadError: '加载 API 密钥失败',
      createNew: '创建新密钥',
      description: '描述',
      descriptionPlaceholder: '密钥描述（可选）',
      createKey: '创建密钥',
      keyGenerated: '新密钥已生成（请保存，仅显示一次）',
      saveWarning: '请立即保存此密钥，它只会显示一次！',
      existingKeys: '现有密钥',
      noKeys: '暂无 API 密钥',
      noDescription: '无描述',
      created: '创建时间',
      delete: '删除',
      deleteConfirm: '确定要删除此 API 密钥吗？',
      status: {
        active: '活跃',
        inactive: '非活跃'
      }
    },

    // Common
    common: {
      chooseFile: '选择文件',
      fileSelected: '已选择文件',
      noFileSelected: '未选择任何文件'
    },

    // API Documentation
    api: {
      title: 'API 文档',
      assets: '资产管理',
      models: '模型管理',
      forecast: '预测接口',
      anomaly: '异常检测',
      training: '模型训练',
      apiKeys: 'API 密钥',
      method: '方法',
      endpoint: '接口地址',
      description: '描述',
      parameters: '参数',
      response: '响应',
      example: '示例',
      required: '必需',
      optional: '可选',
      getAssetList: '获取资产列表',
      createAsset: '创建新资产',
      updateAsset: '更新资产信息',
      deleteAsset: '删除资产',
      getModelList: '获取模型列表',
      getModelDetails: '获取模型详情',
      deleteModel: '删除模型',
      predictFromCsv: '基于CSV文件进行能耗预测',
      predictFromS3: '基于S3数据进行能耗预测',
      detectAnomalyFromCsv: '基于CSV文件进行异常检测',
      detectAnomalyFromS3: '基于S3数据进行异常检测',
      startTraining: '启动新的模型训练任务',
      startTrainingFromCsv: '使用CSV文件启动模型训练任务',
      getTaskStatus: '查询训练任务状态',
      getAssetModels: '获取指定资产的所有模型',
      getApiKeyList: '获取API密钥列表',
      createApiKey: '创建新的API密钥',
      deleteApiKey: '删除API密钥',
      pathParam: '路径参数',
      formData: '表单数据',
      queryParam: '查询参数',
      csvFile: 'CSV数据文件',
      s3DataPath: 'S3数据路径',
      modelId: '模型ID',
      assetId: '资产ID',
      forecastHorizon: '预测时长（小时）',
      modelType: '模型类型',
      taskDescription: '训练任务描述',
      taskId: '任务ID',
      keyName: 'API密钥名称',
      keyId: 'API密钥ID',
      assetName: '资产名称',
      assetDescription: '资产描述',
      sensitivity: '异常检测敏感度 (0.80-0.99，默认：0.95)',
      successMessage: '操作成功',
      deleteSuccessMessage: '删除成功',
      usage: {
        title: '使用说明',
        authentication: '所有API请求需要在Header中包含 X-API-Key',
        contentType: '请求Content-Type应为 application/json 或 multipart/form-data',
        baseUrl: '基础URL',
        errorHandling: '所有错误响应都包含详细的错误信息和状态码'
      }
    },

    // Common errors
    errors: {
      fetchAssets: '无法加载资产列表',
      fetchModels: '无法加载模型列表',
      selectAssetModel: '请选择资产和模型',
      selectFile: '请选择 CSV 文件',
      enterS3Path: '请输入 S3 路径',
      invalidFile: '请选择有效的 .csv 文件',
      forecastFailed: '预测失败：',
      detectionFailed: '检测失败：',
      unexpected: '发生未知错误'
    }
  },

  en: {
    // Navigation
    nav: {
      home: 'Home',
      forecast: 'Energy Forecast',
      anomaly: 'Anomaly Detection',
      training: 'Model Training',
      models: 'Model Management',
      assets: 'Asset Management',
      apiKeys: 'API Keys',
      apiDocs: 'API Documentation'
    },

    // Home page
    home: {
      title: 'Energy Consumption Prediction & Anomaly Detection System',
      subtitle: 'AI-Powered Industrial Energy Analysis Platform',
      features: {
        prediction: {
          title: 'Smart Prediction',
          desc: 'Multi-algorithm energy consumption forecasting'
        },
        anomaly: {
          title: 'Anomaly Detection',
          desc: 'Real-time identification of energy anomalies'
        },
        async: {
          title: 'Async Processing',
          desc: 'Efficient background task processing'
        },
        management: {
          title: 'Asset Management',
          desc: 'Unified equipment and asset management'
        },
        visualization: {
          title: 'Data Visualization',
          desc: 'Intuitive charts and reports'
        },
        security: {
          title: 'Security & Auth',
          desc: 'API key and permission management'
        }
      },
      getStarted: 'Get Started',
      learnMore: 'Learn More'
    },

    // Training page
    training: {
      title: 'Model Training',
      selectAsset: 'Select Asset',
      chooseAsset: 'Please choose an asset',
      selectModelType: 'Select Model Type',
      selectAlgorithm: 'Select Algorithm',
      chooseAlgorithm: 'Please choose an algorithm',
      uploadTrainingData: 'Upload Training Data',
      algorithms: {
        'LightGBM': {
          label: 'LightGBM',
          desc: 'Lightweight gradient boosting machine, suitable for fast training and high accuracy prediction'
        },
        'TiDE': {
          label: 'TiDE',
          desc: 'Time-series Dense Encoder, designed for long-term forecasting'
        },
        'LSTM': {
          label: 'LSTM',
          desc: 'Long Short-Term Memory network, excels at processing sequence data'
        },
        'TFT': {
          label: 'TFT',
          desc: 'Temporal Fusion Transformer, supports multivariate time series forecasting'
        },
        'TFT (No Past Covariates)': {
          label: 'TFT (No Past Covariates)',
          desc: 'Simplified TFT without using past covariates'
        }
      },
      dataInputMethod: 'Data Input Method',
      uploadCsv: 'Upload CSV File',
      uploadCSV: 'Upload CSV File',
      s3Path: 'S3 Path',
      trainingDataFile: 'Training Data File',
      chooseFile: 'Choose File',
      noFileSelected: 'No file selected',
      fileSelected: 'File Selected',
      s3DataPath: 'S3 Data Path',
      s3Placeholder: 'e.g.: s3://bucket/path/to/data.csv',
      epochs: 'Training Epochs',
      epochsHint: 'Recommended: LightGBM 10-50 epochs, deep learning models 20-100 epochs',
      startTraining: 'Start Training',
      starting: 'Starting...',
      taskRunning: 'Task Running...',
      noAssets: 'No Available Assets',
      errors: {
        failedToLoadAssets: 'Failed to load assets',
        noFile: 'Please select a file',
        noS3Path: 'Please enter S3 path',
        failedToStart: 'Failed to start training'
      },
      status: {
        PENDING: 'Pending',
        PROGRESS: 'Training',
        SUCCESS: 'Training Completed',
        FAILURE: 'Training Failed',
        RETRY: 'Retrying',
        REVOKED: 'Cancelled'
      },
      trainingCompleted: 'Training Completed!',
      trainingFailed: 'Training Failed',
      trainingInProgress: 'Training in Progress',
      statusLabel: 'Status',
      parameters: {
        inputSequenceLength: 'Input Sequence Length',
        outputSequenceLength: 'Output Sequence Length',
        inputSequenceLengthHint: 'Length of historical data for training',
        outputSequenceLengthHint: 'Length of future data to predict',
        title: 'Model Parameter Configuration',
        // Base parameters
        input_chunk_length: 'Input Sequence Length',
        output_chunk_length: 'Output Sequence Length',
        input_chunk_length_desc: 'Length of historical data for prediction (hours)',
        output_chunk_length_desc: 'Length of future time to predict (hours)',
        // Model-specific parameters
        random_state: 'Random Seed',
        random_state_desc: 'Ensures reproducible results',
        hidden_size: 'Hidden Size',
        hidden_size_desc: 'Number of neurons in neural network hidden layers',
        training_length: 'Training Length',
        training_length_desc: 'Sequence length used during training',
        lstm_layers: 'LSTM Layers',
        lstm_layers_desc: 'Number of LSTM layers',
        num_attention_heads: 'Attention Heads',
        num_attention_heads_desc: 'Number of multi-head attention heads',
        dropout: 'Dropout Rate',
        dropout_desc: 'Dropout ratio to prevent overfitting',
        batch_size: 'Batch Size',
        batch_size_desc: 'Number of samples per training batch',
        // Parameter categories
        category: {
          model: 'Model Parameters',
          training: 'Training Parameters',
          data: 'Data Parameters'
        }
      }
    },

    // Forecast page
    forecast: {
      title: 'Energy Forecast',
      selectAsset: 'Select Asset',
      chooseAsset: 'Please choose an asset',
      selectModel: 'Select Model',
      chooseModel: 'Please choose a model',
      forecastHours: 'Forecast Horizon (Hours)',
      trained: 'Trained',
      dataInputMethod: 'Data Input Method',
      uploadCSV: 'Upload CSV File',
      s3Path: 'S3 Path',
      uploadHistoricalData: 'Upload Historical Data CSV',
      s3PathInput: 'S3 Data Path',
      startForecast: 'Start Forecast',
      forecasting: 'Forecasting...',
      results: 'Forecast Results',
      historicalEnergy: 'Historical Energy',
      predictedEnergy: 'Predicted Energy',
      timestamp: 'Timestamp',
      energyKwh: 'Energy (kWh)',
      chartTitle: 'Energy Forecast Chart',
      fileSelected: 'Selected',
      lastSelectedFile: 'Last selected file',
      reselect: 'please reselect',
      waitingResults: 'Waiting for forecast results...',
      configureAndPredict: 'Configure parameters and click forecast to display results here',
      visualAnalysis: 'Supports visualization analysis of historical and forecast data'
    },

    // Anomaly Detection page
    anomaly: {
      title: 'Anomaly Detection',
      selectAsset: 'Select Asset',
      selectModel: 'Select Model',
      trained: 'Trained',
      sensitivity: 'Anomaly Detection Sensitivity',
      moreAnomalies: 'More Anomalies',
      fewerAnomalies: 'Fewer Anomalies',
      sensitivityDesc: 'Adjust the strictness of anomaly detection, higher values mean stricter detection',
      dataInputMethod: 'Data Input Method',
      uploadCSV: 'Upload CSV File',
      s3Path: 'S3 Path',
      uploadHistoricalData: 'Upload Historical Data CSV',
      s3PathInput: 'S3 Data Path',
      startDetection: 'Start Detection',
      detecting: 'Detecting...',
      noDetectors: 'No anomaly detectors available for this asset',
      noModels: 'No models with detectors available',
      historicalEnergy: 'Historical Energy',
      anomalies: 'Anomalies',
      timestamp: 'Timestamp',
      energyKwh: 'Energy (kWh)',
      chartTitle: 'Anomaly Detection Results'
    },

    // Models page
    models: {
      title: 'Model Management',
      loading: 'Loading models...',
      refreshing: 'Refreshing...',
      refresh: 'Refresh List',
      loadError: 'Failed to load models',
      noModels: 'No models found',
      startTraining: 'You can start a new training job on the "Model Training" page',
      modelId: 'Model ID',
      asset: 'Asset',
      type: 'Type',
      version: 'Version',
      created: 'Created',
      mape: 'MAPE Error',
      s3Path: 'S3 Path',
      status: {
        completed: 'Completed',
        training: 'Training',
        pending: 'Pending',
        failed: 'Failed'
      },
      trainingParameters: 'Training Parameter Configuration',
      parameterType: 'Type'
    },

    // Assets page
    assets: {
      title: 'Asset Management',
      loading: 'Loading assets...',
      fetchError: 'Failed to fetch assets',
      saveError: 'Failed to save asset',
      deleteError: 'Failed to delete asset',
      createNew: 'Create New Asset',
      editAsset: 'Edit Asset',
      createAsset: 'Create Asset',
      id: 'ID',
      name: 'Name',
      description: 'Description',
      modelCount: 'Model Count',
      actions: 'Actions',
      assetId: 'Asset ID',
      assetName: 'Asset Name',
      assetDescription: 'Asset Description',
      idCannotChange: 'ID cannot be changed after creation',
      cancel: 'Cancel',
      create: 'Create',
      saveChanges: 'Save Changes',
      deleteConfirm: 'Are you sure you want to delete asset {id}? This cannot be undone.'
    },

    // API Keys page
    apiKeys: {
      title: 'API Key Management',
      loading: 'Loading API keys...',
      loadError: 'Failed to load API keys',
      createNew: 'Create New Key',
      description: 'Description',
      descriptionPlaceholder: 'Key description (optional)',
      createKey: 'Create Key',
      keyGenerated: 'New key generated (please save, shown only once)',
      saveWarning: 'Please save this key immediately, it will only be shown once!',
      existingKeys: 'Existing Keys',
      noKeys: 'No API keys found',
      noDescription: 'No description',
      created: 'Created',
      delete: 'Delete',
      deleteConfirm: 'Are you sure you want to delete this API key?',
      status: {
        active: 'Active',
        inactive: 'Inactive'
      }
    },

    // Common
    common: {
      chooseFile: 'Choose File',
      fileSelected: 'File Selected',
      noFileSelected: 'No file selected'
    },

    // API Documentation
    api: {
      title: 'API Documentation',
      assets: 'Asset Management',
      models: 'Model Management',
      forecast: 'Forecast API',
      anomaly: 'Anomaly Detection',
      training: 'Model Training',
      apiKeys: 'API Keys',
      method: 'Method',
      endpoint: 'Endpoint',
      description: 'Description',
      parameters: 'Parameters',
      response: 'Response',
      example: 'Example',
      required: 'Required',
      optional: 'Optional',
      getAssetList: 'Get asset list',
      createAsset: 'Create new asset',
      updateAsset: 'Update asset information',
      deleteAsset: 'Delete asset',
      getModelList: 'Get model list',
      getModelDetails: 'Get model details',
      deleteModel: 'Delete model',
      predictFromCsv: 'Energy prediction from CSV file',
      predictFromS3: 'Energy prediction from S3 data',
      detectAnomalyFromCsv: 'Anomaly detection from CSV file',
      detectAnomalyFromS3: 'Anomaly detection from S3 data',
      startTraining: 'Start new model training task',
      startTrainingFromCsv: 'Start model training from CSV file',
      getTaskStatus: 'Query training task status',
      getAssetModels: 'Get all models for specified asset',
      getApiKeyList: 'Get API key list',
      createApiKey: 'Create new API key',
      deleteApiKey: 'Delete API key',
      pathParam: 'Path parameter',
      formData: 'Form data',
      queryParam: 'Query parameter',
      csvFile: 'CSV data file',
      s3DataPath: 'S3 data path',
      modelId: 'Model ID',
      assetId: 'Asset ID',
      forecastHorizon: 'Forecast horizon (hours)',
      modelType: 'Model type',
      taskDescription: 'Training task description',
      taskId: 'Task ID',
      keyName: 'API key name',
      keyId: 'API key ID',
      assetName: 'Asset name',
      assetDescription: 'Asset description',
      sensitivity: 'Anomaly detection sensitivity (0.80-0.99, default: 0.95)',
      successMessage: 'Operation successful',
      deleteSuccessMessage: 'Deleted successfully',
      usage: {
        title: 'Usage Guide',
        authentication: 'All API requests require X-API-Key in the header',
        contentType: 'Request Content-Type should be application/json or multipart/form-data',
        baseUrl: 'Base URL',
        errorHandling: 'All error responses include detailed error messages and status codes'
      }
    },

    // Common errors
    errors: {
      fetchAssets: 'Could not load assets',
      fetchModels: 'Could not load models',
      selectAssetModel: 'Please select an asset and model',
      selectFile: 'Please select a CSV file',
      enterS3Path: 'Please enter an S3 path',
      invalidFile: 'Please select a valid .csv file',
      forecastFailed: 'Forecast failed: ',
      detectionFailed: 'Detection failed: ',
      unexpected: 'An unexpected error occurred'
    }
  }
};

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState(() => {
    const saved = localStorage.getItem('language');
    console.log("初始化语言:", saved || 'zh');
    return saved || 'zh';
  });

  useEffect(() => {
    console.log("语言已更改为:", language);
    localStorage.setItem('language', language);
    // 强制重新渲染
    document.title = `能耗预测系统 - ${language === 'zh' ? '中文' : 'English'}`;
  }, [language]);

  const toggleLanguage = () => {
    console.log("切换语言，当前语言:", language);
    const newLanguage = language === 'zh' ? 'en' : 'zh';
    console.log("切换到新语言:", newLanguage);
    setLanguage(newLanguage);
  };

  // 确保t是正确的翻译对象
  const t = translations[language] || translations.zh;

  return (
    <LanguageContext.Provider value={{ language, toggleLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}