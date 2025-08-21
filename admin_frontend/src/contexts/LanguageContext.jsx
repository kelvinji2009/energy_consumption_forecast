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
      apiKeys: 'API 密钥'
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
      title: '开始新的模型训练',
      selectAsset: '选择资产',
      chooseAsset: '请选择资产',
      selectAlgorithm: '选择算法',
      chooseAlgorithm: '请选择算法',
      algorithmDescriptions: {
        LightGBM: '快速梯度提升，适合大数据集',
        LSTM: '长短期记忆网络，适合序列数据',
        TFT: '时间融合变换器，高精度预测',
        TiDE: '时间序列密集编码器，轻量级模型'
      },
      dataInputMethod: '数据输入方式',
      uploadCSV: '上传 CSV 文件',
      s3Path: 'S3 路径',
      trainingDataFile: '训练数据文件',
      chooseFile: '选择文件',
      noFileSelected: '未选择任何文件',
      epochs: '训练轮数',
      epochsHint: '神经网络推荐 20-100 轮，LightGBM 不使用轮数',
      startTraining: '开始训练任务',
      training: '训练中...',
      success: '训练任务创建成功！',
      error: '训练失败'
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
      results: '预测结果'
    },

    // Anomaly Detection page
    anomaly: {
      title: '异常检测',
      selectAsset: '选择资产',
      selectModel: '选择模型',
      trained: '训练时间',
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
      }
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
      apiKeys: 'API Keys'
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
      title: 'Start New Model Training',
      selectAsset: 'Select Asset',
      chooseAsset: 'Please choose an asset',
      selectAlgorithm: 'Select Algorithm',
      chooseAlgorithm: 'Please choose an algorithm',
      algorithmDescriptions: {
        LightGBM: 'Fast gradient boosting, suitable for large datasets',
        LSTM: 'Long Short-Term Memory, suitable for sequence data',
        TFT: 'Temporal Fusion Transformer, high-precision prediction',
        TiDE: 'Time-series Dense Encoder, lightweight model'
      },
      dataInputMethod: 'Data Input Method',
      uploadCSV: 'Upload CSV File',
      s3Path: 'S3 Path',
      trainingDataFile: 'Training Data File',
      chooseFile: 'Choose File',
      noFileSelected: 'No file selected',
      epochs: 'Number of Epochs',
      epochsHint: 'Recommended 20-100 for neural networks, LightGBM does not use epochs',
      startTraining: 'Start Training Job',
      training: 'Training...',
      success: 'Training job created successfully!',
      error: 'Training failed'
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
      results: 'Forecast Results'
    },

    // Anomaly Detection page
    anomaly: {
      title: 'Anomaly Detection',
      selectAsset: 'Select Asset',
      selectModel: 'Select Model',
      trained: 'Trained',
      dataInputMethod: 'Data Input Method',
      uploadCSV: 'Upload CSV File',
      s3Path: 'S3 Path',
      uploadHistoricalData: 'Upload Historical Data CSV',
      s3PathInput: 'S3 Data Path',
      startDetection: 'Start Detection',
      detecting: 'Detecting...',
      noDetectors: 'No models with anomaly detectors found for this asset',
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
      }
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
    return saved || 'zh';
  });

  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  const toggleLanguage = () => {
    setLanguage(prev => prev === 'zh' ? 'en' : 'zh');
  };

  const t = translations[language];

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