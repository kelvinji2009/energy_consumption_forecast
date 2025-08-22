import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

function ApiDocumentation() {
  const { t, language } = useLanguage();
  const [selectedCategory, setSelectedCategory] = useState('assets');

  // 改进的翻译函数，确保能够正确获取当前语言的翻译
  const getText = (key) => {
    if (!t || typeof t !== 'object') {
      return key;
    }
    
    // 支持嵌套键，如 'api.title'
    const keys = key.split('.');
    let result = t;
    for (const k of keys) {
      if (result && typeof result === 'object' && k in result) {
        result = result[k];
      } else {
        return key;
      }
    }
    
    return result || key;
  };

  const apiCategories = {
    assets: {
      title: getText('api.assets'),
      icon: '🏭',
      endpoints: [
        {
          method: 'GET',
          path: '/admin/assets',
          description: getText('api.getAssetList'),
          parameters: [],
          response: {
            type: 'array',
            example: [
              {
                id: 'production_line_A',
                name: language === 'zh' ? '生产线 A' : 'Production Line A',
                description: language === 'zh' ? '主要生产线设备' : 'Main production line equipment',
                model_count: 2,
                created_at: '2024-01-01T00:00:00Z'
              }
            ]
          }
        },
        {
          method: 'POST',
          path: '/admin/assets',
          description: getText('api.createAsset'),
          parameters: [
            { name: 'id', type: 'string', required: true, description: getText('api.assetId') },
            { name: 'name', type: 'string', required: true, description: getText('api.assetName') },
            { name: 'description', type: 'string', required: false, description: getText('api.assetDescription') }
          ],
          response: {
            type: 'object',
            example: {
              id: 'production_line_B',
              name: language === 'zh' ? '生产线 B' : 'Production Line B',
              description: language === 'zh' ? '新建生产线设备' : 'New production line equipment',
              created_at: '2024-01-01T00:00:00Z'
            }
          }
        },
        {
          method: 'PUT',
          path: '/admin/assets/{asset_id}',
          description: getText('api.updateAsset'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'name', type: 'string', required: false, description: getText('api.assetName') },
            { name: 'description', type: 'string', required: false, description: getText('api.assetDescription') }
          ],
          response: {
            type: 'object',
            example: {
              id: 'production_line_A',
              name: language === 'zh' ? '更新后的生产线 A' : 'Updated Production Line A',
              description: language === 'zh' ? '更新后的描述' : 'Updated description'
            }
          }
        },
        {
          method: 'DELETE',
          path: '/admin/assets/{asset_id}',
          description: getText('api.deleteAsset'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: { message: getText('api.deleteSuccessMessage') }
          }
        }
      ]
    },
    models: {
      title: getText('api.models'),
      icon: '🤖',
      endpoints: [
        {
          method: 'GET',
          path: '/admin/models',
          description: getText('api.getModelList'),
          parameters: [
            { name: 'asset_id', type: 'string', required: false, description: `${language === 'zh' ? '按' : 'Filter by '}${getText('api.assetId')}${language === 'zh' ? '筛选' : ''}` }
          ],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                asset_id: 'production_line_A',
                model_type: 'TFT',
                version: 1,
                status: 'COMPLETED',
                mape: 0.05,
                created_at: '2024-01-01T00:00:00Z'
              }
            ]
          }
        },
        {
          method: 'GET',
          path: '/admin/models/{model_id}',
          description: getText('api.getModelDetails'),
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: `${getText('api.modelId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: {
              id: 1,
              asset_id: 'production_line_A',
              model_type: 'TFT',
              version: 1,
              status: 'COMPLETED',
              mape: 0.05,
              model_path: 's3://bucket/models/model_1.pkl',
              created_at: '2024-01-01T00:00:00Z'
            }
          }
        },
        {
          method: 'GET',
          path: '/admin/assets/{asset_id}/models',
          description: getText('api.getAssetModels'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                asset_id: 'production_line_A',
                model_type: 'LightGBM',
                model_version: '20250821032849',
                status: 'COMPLETED',
                description: language === 'zh' ? 'LightGBM模型，训练于2025-08-21' : 'LightGBM model trained on 2025-08-21',
                metrics: { mape: 9.324668638609689 },
                detector_path: 'production_line_A/1_20250821032849/detector.joblib',
                created_at: '2025-08-21T03:28:11.193719Z'
              }
            ]
          }
        }
      ]
    },
    forecast: {
      title: getText('api.forecast'),
      icon: '📊',
      endpoints: [
        {
          method: 'POST',
          path: '/assets/{asset_id}/predict',
          description: language === 'zh' ? '基于JSON数据进行预测' : 'Predict based on JSON data',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'historical_data', type: 'array', required: true, description: language === 'zh' ? '历史时间序列数据' : 'Historical time series data' },
            { name: 'forecast_horizon', type: 'integer', required: true, description: getText('api.forecastHorizon') }
          ],
          response: {
            type: 'object',
            example: {
              asset_id: 'production_line_A',
              forecast_data: [
                { timestamp: '2024-01-02T00:00:00Z', predicted_value: 105.2 }
              ]
            }
          }
        },
        {
          method: 'POST',
          path: '/assets/{asset_id}/predict_from_csv',
          description: getText('api.predictFromCsv'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'file', type: 'file', required: true, description: getText('api.csvFile') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') },
            { name: 'forecast_horizon', type: 'integer', required: false, description: `${getText('api.forecastHorizon')}（${language === 'zh' ? '默认168小时' : 'default 168 hours'}）` }
          ],
          response: {
            type: 'object',
            example: {
              asset_id: 'production_line_A',
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, temp: 25.0, production: 80.0, humidity: 60.0 }
              ],
              forecast_data: [
                { timestamp: '2024-01-02T00:00:00Z', predicted_value: 105.2 }
              ]
            }
          }
        },
        {
          method: 'POST',
          path: '/assets/{asset_id}/predict_from_s3',
          description: getText('api.predictFromS3'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 's3_data_path', type: 'string', required: true, description: getText('api.s3DataPath') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') },
            { name: 'forecast_horizon', type: 'integer', required: false, description: `${getText('api.forecastHorizon')}（${language === 'zh' ? '默认168小时' : 'default 168 hours'}）` }
          ],
          response: {
            type: 'object',
            example: {
              asset_id: 'production_line_A',
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, temp: 25.0, production: 80.0, humidity: 60.0 }
              ],
              forecast_data: [
                { timestamp: '2024-01-02T00:00:00Z', predicted_value: 105.2 }
              ]
            }
          }
        }
      ]
    },
    anomaly: {
      title: getText('api.anomaly'),
      icon: '🚨',
      endpoints: [
        {
          method: 'POST',
          path: '/assets/{asset_id}/detect_anomalies_from_csv',
          description: getText('api.detectAnomalyFromCsv'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'file', type: 'file', required: true, description: getText('api.csvFile') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') }
          ],
          response: {
            type: 'object',
            example: {
              asset_id: 'production_line_A',
              anomalies: [
                { timestamp: '2024-01-01T00:00:00Z', value: 150.5 },
                { timestamp: '2024-01-01T05:00:00Z', value: 200.8 }
              ],
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, temp: 25.0, production: 80.0, humidity: 60.0 }
              ]
            }
          }
        },
        {
          method: 'POST',
          path: '/assets/{asset_id}/detect_anomalies_from_s3',
          description: getText('api.detectAnomalyFromS3'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 's3_data_path', type: 'string', required: true, description: getText('api.s3DataPath') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') }
          ],
          response: {
            type: 'object',
            example: {
              asset_id: 'production_line_A',
              anomalies: [
                { timestamp: '2024-01-01T00:00:00Z', value: 150.5 },
                { timestamp: '2024-01-01T05:00:00Z', value: 200.8 }
              ],
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, temp: 25.0, production: 80.0, humidity: 60.0 }
              ]
            }
          }
        }
      ]
    },
    training: {
      title: getText('api.training'),
      icon: '🎯',
      endpoints: [
        {
          method: 'POST',
          path: '/admin/training-jobs',
          description: getText('api.startTraining'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: getText('api.assetId') },
            { name: 'model_type', type: 'string', required: true, description: `${getText('api.modelType')} (LightGBM, TFT, LSTM, TiDE)` },
            { name: 's3_data_path', type: 'string', required: true, description: language === 'zh' ? 'S3中训练数据文件路径' : 'S3 path to training data file' },
            { name: 'n_epochs', type: 'integer', required: false, description: language === 'zh' ? '训练轮数（默认20）' : 'Number of training epochs (default 20)' },
            { name: 'description', type: 'string', required: false, description: getText('api.taskDescription') },
            { name: 'parameters', type: 'array', required: false, description: language === 'zh' ? '训练参数列表' : 'Training parameters list' }
          ],
          response: {
            type: 'object',
            example: {
              message: language === 'zh' ? '训练任务创建并排队成功' : 'Training job created and queued successfully',
              model_id: 1,
              task_id: 'abc123-def456-ghi789',
              asset_id: 'production_line_A',
              status: 'PENDING'
            }
          }
        },
        {
          method: 'POST',
          path: '/admin/training-jobs-from-csv',
          description: getText('api.startTrainingFromCsv'),
          parameters: [
            { name: 'file', type: 'file', required: true, description: `CSV${language === 'zh' ? '训练数据文件' : ' training data file'}` },
            { name: 'asset_id', type: 'string', required: true, description: getText('api.assetId') },
            { name: 'model_type', type: 'string', required: true, description: `${getText('api.modelType')} (LightGBM, TFT, LSTM, TiDE)` },
            { name: 'n_epochs', type: 'integer', required: false, description: language === 'zh' ? '训练轮数（默认20）' : 'Number of training epochs (default 20)' },
            { name: 'description', type: 'string', required: false, description: getText('api.taskDescription') },
            { name: 'parameters', type: 'string', required: false, description: language === 'zh' ? 'JSON格式的训练参数' : 'Training parameters in JSON format' }
          ],
          response: {
            type: 'object',
            example: {
              message: language === 'zh' ? '从上传CSV创建的训练任务已成功创建并排队' : 'Training job from uploaded CSV created and queued successfully',
              model_id: 1,
              task_id: 'abc123-def456-ghi789',
              asset_id: 'production_line_A',
              status: 'PENDING'
            }
          }
        },
        {
          method: 'GET',
          path: '/admin/tasks/{task_id}/status',
          description: getText('api.getTaskStatus'),
          parameters: [
            { name: 'task_id', type: 'string', required: true, description: `${getText('api.taskId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: {
              task_id: 'abc123-def456-ghi789',
              status: 'COMPLETED',
              result: {
                model_id: 1,
                mape: 0.0932,
                model_path: 'production_line_A/1_20250821032849/model.joblib',
                scaler_path: 'production_line_A/1_20250821032849/scaler.joblib',
                detector_path: 'production_line_A/1_20250821032849/detector.joblib'
              }
            }
          }
        }
      ]
    },
    trainingParameters: {
      title: language === 'zh' ? '训练参数管理' : 'Training Parameters',
      icon: '⚙️',
      endpoints: [
        {
          method: 'POST',
          path: '/admin/training-parameters',
          description: language === 'zh' ? '创建单个训练参数' : 'Create a single training parameter',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') },
            { name: 'parameter_name', type: 'string', required: true, description: language === 'zh' ? '参数名称' : 'Parameter name' },
            { name: 'parameter_value', type: 'string', required: true, description: language === 'zh' ? '参数值' : 'Parameter value' },
            { name: 'parameter_type', type: 'string', required: true, description: language === 'zh' ? '参数类型 (int, float, str, bool)' : 'Parameter type (int, float, str, bool)' },
            { name: 'parameter_category', type: 'string', required: false, description: language === 'zh' ? '参数分类 (model, training, data)' : 'Parameter category (model, training, data)' }
          ],
          response: {
            type: 'object',
            example: {
              id: 1,
              model_id: 1,
              parameter_name: 'n_estimators',
              parameter_value: '100',
              parameter_type: 'int',
              parameter_category: 'model',
              created_at: '2025-08-22T06:00:00Z'
            }
          }
        },
        {
          method: 'POST',
          path: '/admin/training-parameters/batch',
          description: language === 'zh' ? '批量创建训练参数' : 'Create training parameters in batch',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') },
            { name: 'parameters', type: 'array', required: true, description: language === 'zh' ? '参数列表' : 'List of parameters' }
          ],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                model_id: 1,
                parameter_name: 'n_estimators',
                parameter_value: '100',
                parameter_type: 'int',
                parameter_category: 'model',
                created_at: '2025-08-22T06:00:00Z'
              }
            ]
          }
        },
        {
          method: 'GET',
          path: '/admin/training-parameters/{model_id}',
          description: language === 'zh' ? '获取指定模型的训练参数' : 'Get training parameters for a specific model',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: `${getText('api.modelId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                model_id: 1,
                parameter_name: 'n_estimators',
                parameter_value: '100',
                parameter_type: 'int',
                parameter_category: 'model',
                created_at: '2025-08-22T06:00:00Z'
              }
            ]
          }
        },
        {
          method: 'DELETE',
          path: '/admin/training-parameters/{model_id}',
          description: language === 'zh' ? '删除指定模型的所有训练参数' : 'Delete all training parameters for a specific model',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: `${getText('api.modelId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: { message: getText('api.deleteSuccessMessage') }
          }
        }
      ]
    },
    apiKeys: {
      title: getText('api.apiKeys'),
      icon: '🔑',
      endpoints: [
        {
          method: 'GET',
          path: '/admin/api-keys',
          description: getText('api.getApiKeyList'),
          parameters: [],
          response: {
            type: 'array',
            example: [
              {
                id: 'f47ac10b-58cc-4372-a567-0e02b2c3d479',
                key_hash: '$2b$12$...',
                description: language === 'zh' ? '生产环境密钥' : 'Production Key',
                is_active: true,
                created_at: '2024-01-01T00:00:00Z',
                expires_at: null
              }
            ]
          }
        },
        {
          method: 'POST',
          path: '/admin/api-keys',
          description: getText('api.createApiKey'),
          parameters: [
            { name: 'description', type: 'string', required: false, description: language === 'zh' ? 'API密钥描述' : 'API key description' }
          ],
          response: {
            type: 'object',
            example: {
              id: 'f47ac10b-58cc-4372-a567-0e02b2c3d479',
              key_hash: '$2b$12$...',
              key: 'f47ac10b-58cc-4372-a567-0e02b2c3d479',
              description: language === 'zh' ? '生产环境密钥' : 'Production Key',
              is_active: true,
              created_at: '2024-01-01T00:00:00Z',
              expires_at: null
            }
          }
        },
        {
          method: 'DELETE',
          path: '/admin/api-keys/{key_id}',
          description: getText('api.deleteApiKey'),
          parameters: [
            { name: 'key_id', type: 'string', required: true, description: `${language === 'zh' ? 'API密钥UUID' : 'API key UUID'}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: { message: getText('api.deleteSuccessMessage') }
          }
        }
      ]
    },
    system: {
      title: language === 'zh' ? '系统状态' : 'System Status',
      icon: '🔧',
      endpoints: [
        {
          method: 'GET',
          path: '/ping',
          description: language === 'zh' ? '检查服务状态' : 'Check service status',
          parameters: [],
          response: {
            type: 'object',
            example: {
              status: 'ok'
            }
          }
        }
      ]
    }
  };

  const getMethodColor = (method) => {
    const colors = {
      GET: '#10b981',
      POST: '#3b82f6',
      PUT: '#f59e0b',
      DELETE: '#ef4444'
    };
    return colors[method] || '#6b7280';
  };

  const renderEndpoint = (endpoint, index) => (
    <div key={index} style={{
      background: 'white',
      borderRadius: '12px',
      padding: '1.5rem',
      marginBottom: '1rem',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
      border: '1px solid #e5e7eb'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        marginBottom: '1rem'
      }}>
        <span style={{
          background: getMethodColor(endpoint.method),
          color: 'white',
          padding: '0.25rem 0.75rem',
          borderRadius: '6px',
          fontSize: '0.875rem',
          fontWeight: '600'
        }}>
          {endpoint.method}
        </span>
        <code style={{
          background: '#f3f4f6',
          padding: '0.5rem',
          borderRadius: '6px',
          fontSize: '0.875rem',
          fontFamily: 'monospace'
        }}>
          {endpoint.path}
        </code>
      </div>

      <p style={{
        color: '#4b5563',
        marginBottom: '1rem',
        fontSize: '0.95rem'
      }}>
        {endpoint.description}
      </p>

      {endpoint.parameters && endpoint.parameters.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <h4 style={{
            color: '#374151',
            fontSize: '0.9rem',
            fontWeight: '600',
            marginBottom: '0.5rem'
          }}>
            📋 {getText('api.parameters')}:
          </h4>
          <div style={{
            background: '#f9fafb',
            borderRadius: '8px',
            padding: '1rem'
          }}>
            {endpoint.parameters.map((param, paramIndex) => (
              <div key={paramIndex} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '0.5rem',
                fontSize: '0.875rem'
              }}>
                <code style={{
                  background: '#e5e7eb',
                  padding: '0.25rem 0.5rem',
                  borderRadius: '4px',
                  fontWeight: '600'
                }}>
                  {param.name}
                </code>
                <span style={{
                  color: '#6b7280',
                  fontSize: '0.8rem'
                }}>
                  {param.type}
                </span>
                {param.required && (
                  <span style={{
                    background: '#fef2f2',
                    color: '#dc2626',
                    padding: '0.125rem 0.375rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem'
                  }}>
                    {getText('api.required')}
                  </span>
                )}
                <span style={{ color: '#4b5563' }}>
                  - {param.description}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h4 style={{
          color: '#374151',
          fontSize: '0.9rem',
          fontWeight: '600',
          marginBottom: '0.5rem'
        }}>
          📤 {getText('api.response')} {getText('api.example')}:
        </h4>
        <pre style={{
          background: '#1f2937',
          color: '#f9fafb',
          padding: '1rem',
          borderRadius: '8px',
          fontSize: '0.8rem',
          overflow: 'auto',
          fontFamily: 'monospace'
        }}>
          {JSON.stringify(endpoint.response.example, null, 2)}
        </pre>
      </div>
    </div>
  );

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderRadius: '20px',
        padding: '2rem',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <h2 style={{
          color: '#4a5568',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          📚 {getText('api.title')}
        </h2>

        <div style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '2rem',
          flexWrap: 'wrap'
        }}>
          {Object.entries(apiCategories).map(([key, category]) => (
            <button
              key={key}
              onClick={() => setSelectedCategory(key)}
              style={{
                padding: '0.75rem 1.5rem',
                background: selectedCategory === key 
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : 'rgba(255, 255, 255, 0.8)',
                color: selectedCategory === key ? 'white' : '#4a5568',
                border: '2px solid',
                borderColor: selectedCategory === key 
                  ? 'transparent'
                  : 'rgba(255, 255, 255, 0.3)',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.9rem',
                fontWeight: '500',
                boxShadow: selectedCategory === key 
                  ? '0 4px 15px rgba(102, 126, 234, 0.4)'
                  : '0 2px 8px rgba(0, 0, 0, 0.1)'
              }}
              onMouseOver={(e) => {
                if (selectedCategory !== key) {
                  e.target.style.background = 'rgba(255, 255, 255, 0.9)';
                  e.target.style.transform = 'translateY(-2px)';
                }
              }}
              onMouseOut={(e) => {
                if (selectedCategory !== key) {
                  e.target.style.background = 'rgba(255, 255, 255, 0.8)';
                  e.target.style.transform = 'translateY(0)';
                }
              }}
            >
              <span>{category.icon}</span>
              {category.title}
            </button>
          ))}
        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.6)',
          borderRadius: '16px',
          padding: '2rem',
          backdropFilter: 'blur(5px)'
        }}>
          <h3 style={{
            color: '#374151',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '1.25rem',
            fontWeight: '600'
          }}>
            {apiCategories[selectedCategory].icon} {apiCategories[selectedCategory].title}
          </h3>

          {apiCategories[selectedCategory].endpoints.map((endpoint, index) => 
            renderEndpoint(endpoint, index)
          )}
        </div>

        <div style={{
          marginTop: '2rem',
          padding: '1.5rem',
          background: 'rgba(59, 130, 246, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(59, 130, 246, 0.2)'
        }}>
          <h4 style={{
            color: '#1e40af',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '1.1rem',
            fontWeight: '600'
          }}>
            💡 {getText('api.usage.title')}
          </h4>
          <div style={{
            color: '#1e3a8a',
            fontSize: '0.9rem',
            lineHeight: '1.6'
          }}>
            <p style={{ marginBottom: '0.5rem' }}>
              • {getText('api.usage.authentication')}
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              • {getText('api.usage.contentType')}
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              • {getText('api.usage.baseUrl')}: <code style={{
                background: 'rgba(59, 130, 246, 0.1)',
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                fontFamily: 'monospace'
              }}>http://localhost:8000</code>
            </p>
            <p>
              • {getText('api.usage.errorHandling')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ApiDocumentation;
