import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

function ApiDocumentation() {
  const { t } = useLanguage();
  const [selectedCategory, setSelectedCategory] = useState('assets');

  // 安全翻译函数
  const safeT = (key) => {
    if (typeof t === 'function') {
      return t(key);
    }
    const defaultTexts = {
      'api.title': 'API 文档',
      'api.assets': '资产管理',
      'api.models': '模型管理',
      'api.forecast': '预测接口',
      'api.anomaly': '异常检测',
      'api.training': '模型训练',
      'api.apiKeys': 'API 密钥',
      'api.method': '方法',
      'api.endpoint': '接口地址',
      'api.description': '描述',
      'api.parameters': '参数',
      'api.response': '响应',
      'api.example': '示例'
    };
    return defaultTexts[key] || key;
  };

  const apiCategories = {
    assets: {
      title: safeT('api.assets'),
      icon: '🏭',
      endpoints: [
      {
        title: '🎯 模型训练',
        apis: [
          {
            method: 'POST',
            endpoint: '/admin/training-jobs',
            description: '启动新的模型训练任务',
            params: [
              { name: 'asset_id', type: 'string', required: true, description: '资产ID' },
              { name: 'model_type', type: 'string', required: true, description: '模型类型 (LightGBM, TFT, LSTM, TiDE)' },
              { name: 'description', type: 'string', required: false, description: '训练任务描述' }
            ],
            response: {
              task_id: 'string',
              status: 'PENDING',
              message: 'Training job started successfully'
            }
          },
          {
            method: 'POST',
            endpoint: '/admin/training-jobs-from-csv',
            description: '使用CSV文件启动模型训练任务',
            params: [
              { name: 'file', type: 'file', required: true, description: 'CSV训练数据文件' },
              { name: 'asset_id', type: 'string', required: true, description: '资产ID' },
              { name: 'model_type', type: 'string', required: true, description: '模型类型 (LightGBM, TFT, LSTM, TiDE)' },
              { name: 'description', type: 'string', required: false, description: '训练任务描述' }
            ],
            response: {
              task_id: 'string',
              status: 'PENDING',
              message: 'Training job started successfully'
            }
          },
          {
            method: 'GET',
            endpoint: '/admin/tasks/{task_id}/status',
            description: '查询训练任务状态',
            params: [
              { name: 'task_id', type: 'string', required: true, description: '任务ID' }
            ],
            response: {
              task_id: 'string',
              status: 'COMPLETED | PENDING | FAILED',
              progress: 'number',
              result: 'object'
            }
          },
          {
            method: 'GET',
            endpoint: '/admin/assets/{asset_id}/models',
            description: '获取指定资产的所有模型',
            params: [
              { name: 'asset_id', type: 'string', required: true, description: '资产ID' }
            ],
            response: [
              {
                id: 'number',
                model_type: 'string',
                model_version: 'string',
                status: 'string',
                description: 'string',
                metrics: 'object',
                created_at: 'datetime'
              }
            ]
          }
        ]
      },
        {
          method: 'PUT',
          path: '/admin/assets/{asset_id}',
          description: '更新资产信息',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' },
            { name: 'name', type: 'string', required: false, description: '资产名称' },
            { name: 'description', type: 'string', required: false, description: '资产描述' }
          ],
          response: {
            type: 'object',
            example: {
              id: 'production_line_A',
              name: '更新后的生产线 A',
              description: '更新后的描述'
            }
          }
        },
        {
          method: 'DELETE',
          path: '/admin/assets/{asset_id}',
          description: '删除资产',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' }
          ],
          response: {
            type: 'object',
            example: { message: '资产删除成功' }
          }
        }
      ]
    },
    models: {
      title: safeT('api.models'),
      icon: '🤖',
      endpoints: [
        {
          method: 'GET',
          path: '/admin/models',
          description: '获取模型列表',
          parameters: [
            { name: 'asset_id', type: 'string', required: false, description: '按资产ID筛选' }
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
          description: '获取模型详情',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: '模型ID（路径参数）' }
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
          description: '删除模型',
          parameters: [
            { name: 'model_id', type: 'integer', required: true, description: '模型ID（路径参数）' }
          ],
          response: {
            type: 'object',
            example: { message: '模型删除成功' }
          }
        }
      ]
    },
    forecast: {
      title: safeT('api.forecast'),
      icon: '📊',
      endpoints: [
        {
          method: 'POST',
          path: '/assets/{asset_id}/predict_from_csv',
          description: '基于CSV文件进行能耗预测',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' },
            { name: 'file', type: 'file', required: true, description: 'CSV数据文件' },
            { name: 'model_id', type: 'integer', required: true, description: '模型ID' },
            { name: 'forecast_horizon', type: 'integer', required: true, description: '预测时长（小时）' }
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
          description: '基于S3数据进行能耗预测',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' },
            { name: 's3_data_path', type: 'string', required: true, description: 'S3数据路径' },
            { name: 'model_id', type: 'integer', required: true, description: '模型ID' },
            { name: 'forecast_horizon', type: 'integer', required: true, description: '预测时长（小时）' }
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
      title: safeT('api.anomaly'),
      icon: '🚨',
      endpoints: [
        {
          method: 'POST',
          path: '/assets/{asset_id}/detect_anomaly_from_csv',
          description: '基于CSV文件进行异常检测',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' },
            { name: 'file', type: 'file', required: true, description: 'CSV数据文件' },
            { name: 'model_id', type: 'integer', required: true, description: '模型ID' }
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
          description: '基于S3数据进行异常检测',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID（路径参数）' },
            { name: 's3_data_path', type: 'string', required: true, description: 'S3数据路径' },
            { name: 'model_id', type: 'integer', required: true, description: '模型ID' }
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
      title: safeT('api.training'),
      icon: '🎯',
      endpoints: [
        {
          method: 'POST',
          path: '/admin/train_model',
          description: '训练新模型',
          parameters: [
            { name: 'asset_id', type: 'string', required: true, description: '资产ID' },
            { name: 'model_type', type: 'string', required: true, description: '模型类型 (TFT, LSTM, LGBM, TIDE)' },
            { name: 'file', type: 'file', required: false, description: '训练数据CSV文件' },
            { name: 's3_data_path', type: 'string', required: false, description: 'S3训练数据路径' }
          ],
          response: {
            type: 'object',
            example: {
              task_id: 'abc123',
              message: '模型训练任务已启动',
              status: 'PENDING'
            }
          }
        },
        {
          method: 'GET',
          path: '/admin/task_status/{task_id}',
          description: '获取训练任务状态',
          parameters: [
            { name: 'task_id', type: 'string', required: true, description: '任务ID（路径参数）' }
          ],
          response: {
            type: 'object',
            example: {
              task_id: 'abc123',
              status: 'SUCCESS',
              result: {
                model_id: 1,
                mape: 0.05
              }
            }
          }
        }
      ]
    },
    apiKeys: {
      title: safeT('api.apiKeys'),
      icon: '🔑',
      endpoints: [
        {
          method: 'GET',
          path: '/admin/api_keys',
          description: '获取API密钥列表',
          parameters: [],
          response: {
            type: 'array',
            example: [
              {
                id: 1,
                name: 'Production Key',
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
          description: '创建新的API密钥',
          parameters: [
            { name: 'name', type: 'string', required: true, description: 'API密钥名称' }
          ],
          response: {
            type: 'object',
            example: {
              id: 1,
              name: 'Production Key',
              key: 'sk-1234567890abcdef',
              created_at: '2024-01-01T00:00:00Z'
            }
          }
        },
        {
          method: 'DELETE',
          path: '/admin/api_keys/{key_id}',
          description: '删除API密钥',
          parameters: [
            { name: 'key_id', type: 'integer', required: true, description: 'API密钥ID（路径参数）' }
          ],
          response: {
            type: 'object',
            example: { message: 'API密钥删除成功' }
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
            📋 {safeT('api.parameters')}:
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
                    必需
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
          📤 {safeT('api.response')} {safeT('api.example')}:
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
          📚 {safeT('api.title')}
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