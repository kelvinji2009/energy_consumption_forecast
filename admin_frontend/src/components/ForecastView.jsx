import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../apiClient';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';
import { useLanguage } from '../contexts/LanguageContext';
import CustomFileInput from './CustomFileInput';

// Register Chart.js components and plugins
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  zoomPlugin
);

function ForecastView() {
  const { t, language } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [forecastHours, setForecastHours] = useState(168);
  const [dataInputMethod, setDataInputMethod] = useState('upload');
  const [file, setFile] = useState(null);
  const [savedFileName, setSavedFileName] = useState('');
  const [s3Path, setS3Path] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [chartData, setChartData] = useState({ datasets: [] });
  const [isInitialized, setIsInitialized] = useState(false);
  const chartRef = useRef(null);

  // 获取翻译文本的函数
  const getText = (key) => {
    if (!key) return '';
    
    // 分解键路径，例如 'forecast.title' => ['forecast', 'title']
    const keys = key.split('.');
    
    // 从t对象中获取翻译
    if (t && typeof t === 'object') {
      let value = t;
      for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
          value = value[k];
        } else {
          // 如果找不到翻译，返回键名
          return key;
        }
      }
      return value;
    }
    
    // 如果t不可用，返回键名
    return key;
  };

  // 监听语言变化，强制重新渲染
  useEffect(() => {
    console.log('Language changed to:', language);
    // 语言变化时重新获取资产列表，触发UI更新
    fetchAssets();
  }, [language]);

  useEffect(() => {
    fetchAssets();
  }, []);

  // 在assets加载完成后恢复状态
  useEffect(() => {
    console.log('Assets loaded:', assets.length);
    if (assets.length > 0) {
      const savedState = localStorage.getItem('forecastViewState');
      console.log('Raw saved state from localStorage:', savedState);
      
      if (savedState) {
        try {
          const state = JSON.parse(savedState);
          console.log('Parsed state:', state);
          console.log('Available assets:', assets.map(a => a.id));
          
          // 恢复资产选择
          if (state.selectedAsset && assets.find(a => a.id === state.selectedAsset)) {
            console.log('Restoring asset:', state.selectedAsset);
            setSelectedAsset(state.selectedAsset);
          } else {
            console.log('Asset not found or not saved:', state.selectedAsset);
          }
          
          // 恢复其他状态 - 只恢复非默认值
          console.log('Restoring other states...');
          if (state.forecastHours && state.forecastHours !== 168) {
            setForecastHours(state.forecastHours);
          }
          if (state.dataInputMethod && state.dataInputMethod !== 'upload') {
            setDataInputMethod(state.dataInputMethod);
          }
          if (state.s3Path) {
            setS3Path(state.s3Path);
          }
          if (state.savedFileName) {
            setSavedFileName(state.savedFileName);
          }
        } catch (e) {
          console.error('Failed to restore saved state:', e);
        }
      } else {
        console.log('No saved state found in localStorage');
      }
      
      // 标记初始化完成
      setIsInitialized(true);
    }
  }, [assets]);

  // 在models加载完成后恢复模型选择
  useEffect(() => {
    if (models.length > 0) {
      const savedState = localStorage.getItem('forecastViewState');
      if (savedState) {
        try {
          const state = JSON.parse(savedState);
          console.log('Trying to restore model:', state.selectedModel, 'type:', typeof state.selectedModel);
          console.log('Available models:', models.map(m => ({ id: m.id, type: typeof m.id, name: m.model_type })));
          
          if (state.selectedModel) {
            console.log('Attempting to restore model:', state.selectedModel);
            // 尝试多种匹配方式
            const foundModel = models.find(m => 
              m.id.toString() === state.selectedModel.toString() || 
              m.id === parseInt(state.selectedModel) ||
              m.id === state.selectedModel
            );
            
            if (foundModel) {
              console.log('Found matching model:', foundModel.id, 'Setting selectedModel to:', foundModel.id.toString());
              // 确保设置的值与模型ID类型一致
              setSelectedModel(foundModel.id.toString());
            } else {
              console.log('No matching model found for:', state.selectedModel);
              console.log('Available model IDs:', models.map(m => m.id));
            }
          } else {
            console.log('No saved model to restore');
          }
        } catch (e) {
          console.error('Failed to restore model state:', e);
        }
      }
    }
  }, [models]);

  useEffect(() => {
    if (selectedAsset) {
      fetchModels(selectedAsset);
    }
  }, [selectedAsset]);

  // 保存状态到localStorage - 只在初始化完成后保存，且不保存空的模型选择
  useEffect(() => {
    if (isInitialized) {
      // 获取当前保存的状态
      const currentSaved = localStorage.getItem('forecastViewState');
      let currentState = {};
      if (currentSaved) {
        try {
          currentState = JSON.parse(currentSaved);
        } catch (e) {
          console.error('Failed to parse current saved state:', e);
        }
      }
      
      const state = {
        selectedAsset,
        selectedModel: selectedModel || currentState.selectedModel || '', // 保留之前的模型选择如果当前为空
        forecastHours,
        dataInputMethod,
        s3Path,
        savedFileName
      };
      
      console.log('Saving state to localStorage (after initialization):', state);
      console.log('Previous saved model:', currentState.selectedModel, 'Current model:', selectedModel);
      
      localStorage.setItem('forecastViewState', JSON.stringify(state));
      
      // 验证保存是否成功
      const saved = localStorage.getItem('forecastViewState');
      console.log('Verified saved state:', saved);
    }
  }, [selectedAsset, selectedModel, forecastHours, dataInputMethod, s3Path, savedFileName, isInitialized]);

  const fetchAssets = async () => {
    try {
      const data = await apiClient('/admin/assets');
      setAssets(data);
    } catch (err) {
      setError(getText('errors.fetchAssets'));
    }
  };

  const fetchModels = async (assetId) => {
    try {
      const data = await apiClient(`/admin/models?asset_id=${assetId}`);
      setModels(data.filter(model => model.status === 'COMPLETED'));
    } catch (err) {
      setError(getText('errors.fetchModels'));
    }
  };

  const handleFileChange = (selectedFile) => {
    setFile(selectedFile);
    if (selectedFile) {
      setSavedFileName(selectedFile.name);
    }
  };

  const handleForecast = async () => {
    if (!selectedAsset || !selectedModel) {
      setError(getText('errors.selectAssetModel'));
      return;
    }

    if (dataInputMethod === 'upload' && !file) {
      setError(getText('errors.selectFile'));
      return;
    }

    if (dataInputMethod === 's3' && !s3Path) {
      setError(getText('errors.enterS3Path'));
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      let endpoint;
      const formData = new FormData();
      formData.append('model_id', selectedModel);
      formData.append('forecast_horizon', forecastHours);

      if (dataInputMethod === 'upload') {
        formData.append('file', file);
        endpoint = `/assets/${selectedAsset}/predict_from_csv`;
      } else {
        // For S3 path, use query parameters instead of form data
        const params = new URLSearchParams({
          s3_data_path: s3Path,
          forecast_horizon: forecastHours,
          model_id: selectedModel
        });
        endpoint = `/assets/${selectedAsset}/predict_from_s3?${params}`;
      }

      const requestOptions = {
        method: 'POST',
        headers: {},
      };

      // Only add body for file upload
      if (dataInputMethod === 'upload') {
        requestOptions.body = formData;
      }

      const result = await apiClient(endpoint, requestOptions);
      
      // Debug: log the actual response structure
      console.log('API Response:', result);
      console.log('Response keys:', Object.keys(result));
      
      // Process data for chart display with safe checks
      let historicalData = [];
      let forecastData = [];
      
      // Handle different possible response structures
      if (result.historical_data && Array.isArray(result.historical_data)) {
        historicalData = result.historical_data
          .filter(d => d && d.timestamp && d.value != null)
          .map(d => ({ x: new Date(d.timestamp), y: d.value }));
      }
      
      // API returns forecast_data, not forecast
      if (result.forecast_data && Array.isArray(result.forecast_data)) {
        forecastData = result.forecast_data
          .filter(d => d && d.timestamp && d.predicted_value != null)
          .map(d => ({ x: new Date(d.timestamp), y: d.predicted_value }));
      } else if (result.forecast && Array.isArray(result.forecast)) {
        // Fallback for different structure
        forecastData = result.forecast
          .filter(d => d && d.timestamp && d.value != null)
          .map(d => ({ x: new Date(d.timestamp), y: d.value }));
      }

      setChartData({
        datasets: [
          {
            label: getText('forecast.historicalEnergy'),
            data: historicalData,
            borderColor: '#8884d8',
            backgroundColor: 'rgba(136, 132, 216, 0.5)',
            pointRadius: 1,
            type: 'line',
          },
          {
            label: getText('forecast.predictedEnergy'),
            data: forecastData,
            borderColor: '#82ca9d',
            backgroundColor: 'rgba(130, 202, 157, 0.5)',
            pointRadius: 2,
            type: 'line',
            borderDash: [5, 5],
          }
        ]
      });
    } catch (err) {
      console.error('Forecast error:', err);
      let errorMessage = getText('errors.unexpected');
      
      if (err.message) {
        errorMessage = err.message;
      } else if (typeof err === 'string') {
        errorMessage = err;
      } else if (err.detail) {
        errorMessage = err.detail;
      } else {
        errorMessage = JSON.stringify(err);
      }
      
      setError(`${getText('errors.forecastFailed')}${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

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
          📈 {getText('forecast.title')}
        </h2>

        <div style={{ display: 'grid', gap: '1.5rem' }}>
          <div>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              🏭 {getText('forecast.selectAsset')}:
            </label>
            <select
              value={selectedAsset}
              onChange={e => setSelectedAsset(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: 'white',
                transition: 'all 0.3s ease'
              }}
            >
              <option value="">{getText('forecast.chooseAsset')}</option>
              {assets.map(asset => (
                <option key={asset.id} value={asset.id}>
                  {asset.name} ({asset.id})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              🤖 {getText('forecast.selectModel')}:
            </label>
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              disabled={!selectedAsset || models.length === 0}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: selectedAsset ? 'white' : '#f7fafc',
                transition: 'all 0.3s ease'
              }}
            >
              <option value="">{getText('forecast.chooseModel')}</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  v{model.version} - {model.model_type} | MAPE: {model.mape ? `${(model.mape * 100).toFixed(2)}%` : 'N/A'} | {getText('forecast.trained')}: {new Date(model.created_at).toLocaleDateString()} (ID: {model.id})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              ⏱️ {getText('forecast.forecastHours')}:
            </label>
            <input
              type="number"
              value={forecastHours}
              onChange={e => setForecastHours(parseInt(e.target.value))}
              min="1"
              max="8760"
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: 'white',
                transition: 'all 0.3s ease'
              }}
            />
          </div>

          <div>
            <label style={{
              display: 'block',
              marginBottom: '1rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              📁 {getText('forecast.dataInputMethod')}:
            </label>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1rem',
                background: dataInputMethod === 'upload' ? '#667eea' : 'rgba(255, 255, 255, 0.8)',
                color: dataInputMethod === 'upload' ? 'white' : '#4a5568',
                borderRadius: '25px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                border: '2px solid',
                borderColor: dataInputMethod === 'upload' ? '#667eea' : '#e2e8f0'
              }}>
                <input
                  type="radio"
                  value="upload"
                  checked={dataInputMethod === 'upload'}
                  onChange={() => setDataInputMethod('upload')}
                  style={{ display: 'none' }}
                />
                📤 {getText('forecast.uploadCSV')}
              </label>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1rem',
                background: dataInputMethod === 's3' ? '#667eea' : 'rgba(255, 255, 255, 0.8)',
                color: dataInputMethod === 's3' ? 'white' : '#4a5568',
                borderRadius: '25px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                border: '2px solid',
                borderColor: dataInputMethod === 's3' ? '#667eea' : '#e2e8f0'
              }}>
                <input
                  type="radio"
                  value="s3"
                  checked={dataInputMethod === 's3'}
                  onChange={() => setDataInputMethod('s3')}
                  style={{ display: 'none' }}
                />
                ☁️ {getText('forecast.s3Path')}
              </label>
            </div>

            {dataInputMethod === 'upload' ? (
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '0.5rem',
                  fontWeight: '500',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {getText('forecast.uploadHistoricalData')}:
                </label>
                <CustomFileInput
                  onFileChange={handleFileChange}
                  accept=".csv"
                  selectedFile={file}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px dashed #cbd5e0',
                    borderRadius: '12px',
                    background: '#f7fafc',
                    cursor: 'pointer'
                  }}
                />
                {!file && savedFileName && (
                  <p style={{ marginTop: '0.5rem', color: '#3182ce', fontSize: '0.9rem' }}>
                    💡 {getText('forecast.lastSelectedFile')}: {savedFileName} ({getText('forecast.reselect')})
                  </p>
                )}
              </div>
            ) : (
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '0.5rem',
                  fontWeight: '500',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {getText('forecast.s3PathInput')}:
                </label>
                <input
                  type="text"
                  value={s3Path}
                  onChange={e => setS3Path(e.target.value)}
                  placeholder="s3://bucket/path/to/data.csv"
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px solid #e2e8f0',
                    borderRadius: '12px',
                    fontSize: '1rem',
                    background: 'white'
                  }}
                />
              </div>
            )}
          </div>

          {error && (
            <div style={{
              padding: '1rem',
              background: 'rgba(254, 178, 178, 0.9)',
              color: '#c53030',
              borderRadius: '12px',
              border: '1px solid #feb2b2'
            }}>
              {error}
            </div>
          )}

          <button
            onClick={handleForecast}
            disabled={isLoading || !assets.length || !models.length}
            style={{
              padding: '1rem 2rem',
              background: isLoading ? '#a0aec0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '25px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            {isLoading ? '⏳' : '▶️'} {isLoading ? getText('forecast.forecasting') : getText('forecast.startForecast')}
          </button>
        </div>

        <div style={{ position: 'relative', width: '100%', height: '400px', marginTop: '2rem' }}>
          <Line 
            ref={chartRef} 
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  type: 'time',
                  time: {
                    unit: 'day',
                    tooltipFormat: 'MMM dd, yyyy HH:mm',
                  },
                  title: {
                    display: true,
                    text: getText('forecast.timestamp')
                  }
                },
                y: {
                  title: {
                    display: true,
                    text: getText('forecast.energyKwh')
                  }
                }
              },
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: getText('forecast.chartTitle')
                },
                zoom: {
                  pan: {
                    enabled: true,
                    mode: 'x',
                  },
                  zoom: {
                    wheel: {
                      enabled: true,
                    },
                    pinch: {
                      enabled: true
                    },
                    mode: 'x',
                  }
                }
              },
              animation: false,
            }} 
            data={chartData.datasets.length > 0 ? chartData : {
              datasets: [{
                label: getText('forecast.waitingResults'),
                data: [],
                borderColor: '#e2e8f0',
                backgroundColor: 'rgba(226, 232, 240, 0.1)',
                pointRadius: 0,
                type: 'line',
              }]
            }} 
          />
          {chartData.datasets.length === 0 && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              color: '#a0aec0',
              pointerEvents: 'none'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
              <div style={{ fontSize: '1.2rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                {getText('forecast.configureAndPredict')}
              </div>
              <div style={{ fontSize: '0.9rem', color: '#718096' }}>
                {getText('forecast.visualAnalysis')}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ForecastView;