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
          method: 'DELETE',
          path: '/admin/models/{model_id}',
          description: getText('api.deleteModel'),
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
    forecast: {
      title: getText('api.forecast'),
      icon: '📊',
      endpoints: [
        {
          method: 'POST',
          path: '/assets/{asset_id}/predict_from_csv',
          description: getText('api.predictFromCsv'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'file', type: 'file', required: true, description: getText('api.csvFile') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') },
            { name: 'forecast_horizon', type: 'integer', required: true, description: getText('api.forecastHorizon') }
          ],
          response: {
            type: 'object',
            example: {
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5 }
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
            { name: 'forecast_horizon', type: 'integer', required: true, description: getText('api.forecastHorizon') }
          ],
          response: {
            type: 'object',
            example: {
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5 }
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
          path: '/assets/{asset_id}/detect_anomaly_from_csv',
          description: getText('api.detectAnomalyFromCsv'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 'file', type: 'file', required: true, description: getText('api.csvFile') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') }
          ],
          response: {
            type: 'object',
            example: {
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, is_anomaly: false }
              ],
              anomaly_summary: {
                total_points: 1000,
                anomaly_count: 15,
                anomaly_rate: 0.015
              }
            }
          }
        },
        {
          method: 'POST',
          path: '/assets/{asset_id}/detect_anomaly_from_s3',
          description: getText('api.detectAnomalyFromS3'),
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: `${getText('api.assetId')}（${getText('api.pathParam')}）` },
            { name: 's3_data_path', type: 'string', required: true, description: getText('api.s3DataPath') },
            { name: 'model_id', type: 'integer', required: true, description: getText('api.modelId') }
          ],
          response: {
            type: 'object',
            example: {
              historical_data: [
                { timestamp: '2024-01-01T00:00:00Z', value: 100.5, is_anomaly: false }
              ],
              anomaly_summary: {
                total_points: 1000,
                anomaly_count: 15,
                anomaly_rate: 0.015
              }
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
            { name: 'description', type: 'string', required: false, description: getText('api.taskDescription') }
          ],
          response: {
            type: 'object',
            example: {
              task_id: 'abc123-def456-ghi789',
              status: 'PENDING',
              message: language === 'zh' ? '训练任务启动成功' : 'Training job started successfully'
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
            { name: 'description', type: 'string', required: false, description: getText('api.taskDescription') }
          ],
          response: {
            type: 'object',
            example: {
              task_id: 'abc123-def456-ghi789',
              status: 'PENDING',
              message: language === 'zh' ? '训练任务启动成功' : 'Training job started successfully'
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
              progress: 100,
              result: {
                model_id: 1,
                mape: 0.0932,
                model_path: 'production_line_A/1_20250821032849/model.joblib'
              }
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
                created_at: '2025-08-21T03:28:11.193719Z'
              }
            ]
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
          path: '/admin/api_keys',
          description: getText('api.getApiKeyList'),
          parameters: [],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                name: language === 'zh' ? '生产环境密钥' : 'Production Key',
                key_preview: 'sk-...abc123',
                created_at: '2024-01-01T00:00:00Z',
                last_used: '2024-01-02T00:00:00Z'
              }
            ]
          }
        },
        {
          method: 'POST',
          path: '/admin/api_keys',
          description: getText('api.createApiKey'),
          parameters: [
            { name: 'name', type: 'string', required: true, description: getText('api.keyName') }
          ],
          response: {
            type: 'object',
            example: {
              id: 1,
              name: language === 'zh' ? '生产环境密钥' : 'Production Key',
              key: 'sk-1234567890abcdef',
              created_at: '2024-01-01T00:00:00Z'
            }
          }
        },
        {
          method: 'DELETE',
          path: '/admin/api_keys/{key_id}',
          description: getText('api.deleteApiKey'),
          parameters: [
            { name: 'key_id', type: 'integer', required: true, description: `${getText('api.keyId')}（${getText('api.pathParam')}）` }
          ],
          response: {
            type: 'object',
            example: { message: getText('api.deleteSuccessMessage') }
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
                borderColor: selectedCategory === key ? '#667eea' : '#e2e8f0',
                borderRadius: '25px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                fontSize: '0.9rem',
                fontWeight: '500',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              {category.icon} {category.title}
            </button>
          ))}
        </div>

        <div>
          <h3 style={{
            color: '#374151',
            fontSize: '1.25rem',
            fontWeight: '600',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            {apiCategories[selectedCategory].icon} {apiCategories[selectedCategory].title}
          </h3>

          {apiCategories[selectedCategory].endpoints.map((endpoint, index) => 
            renderEndpoint(endpoint, index)
          )}
        </div>
      </div>
    </div>
  );
}

export default ApiDocumentation;