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
  const { t } = useLanguage();
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

  // 安全翻译函数
  const safeT = (key) => {
    if (typeof t === 'function') {
      return t(key);
    }
    // 提供默认的中文文本
    const defaultTexts = {
      'forecast.title': '能耗预测',
      'forecast.selectAsset': '选择资产',
      'forecast.chooseAsset': '请选择资产',
      'forecast.selectModel': '选择模型',
      'forecast.chooseModel': '请选择模型',
      'forecast.trained': '训练时间',
      'forecast.forecastHours': '预测步长（小时）',
      'forecast.dataInputMethod': '数据输入方式',
      'forecast.uploadCSV': '上传 CSV 文件',
      'forecast.s3Path': 'S3 路径',
      'forecast.uploadHistoricalData': '上传历史数据 CSV',
      'forecast.s3PathInput': 'S3 数据路径',
      'forecast.forecasting': '预测中...',
      'forecast.startForecast': '开始预测',
      'forecast.results': '预测结果',
      'errors.fetchAssets': '获取资产失败',
      'errors.fetchModels': '获取模型失败',
      'errors.selectAssetModel': '请选择资产和模型',
      'errors.selectFile': '请选择文件',
      'errors.enterS3Path': '请输入S3路径',
      'errors.forecastFailed': '预测失败：',
      'errors.unexpected': '发生了意外错误'
    };
    return defaultTexts[key] || key;
  };

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
      setError(safeT('errors.fetchAssets'));
    }
  };

  const fetchModels = async (assetId) => {
    try {
      const data = await apiClient(`/admin/models?asset_id=${assetId}`);
      setModels(data.filter(model => model.status === 'COMPLETED'));
    } catch (err) {
      setError(safeT('errors.fetchModels'));
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setSavedFileName(selectedFile.name);
    }
  };

  const handleForecast = async () => {
    if (!selectedAsset || !selectedModel) {
      setError(safeT('errors.selectAssetModel'));
      return;
    }

    if (dataInputMethod === 'upload' && !file) {
      setError(safeT('errors.selectFile'));
      return;
    }

    if (dataInputMethod === 's3' && !s3Path) {
      setError(safeT('errors.enterS3Path'));
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
            label: safeT('forecast.historicalEnergy') || '历史能耗',
            data: historicalData,
            borderColor: '#8884d8',
            backgroundColor: 'rgba(136, 132, 216, 0.5)',
            pointRadius: 1,
            type: 'line',
          },
          {
            label: safeT('forecast.predictedEnergy') || '预测能耗',
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
      let errorMessage = safeT('errors.unexpected');
      
      if (err.message) {
        errorMessage = err.message;
      } else if (typeof err === 'string') {
        errorMessage = err;
      } else if (err.detail) {
        errorMessage = err.detail;
      } else {
        errorMessage = JSON.stringify(err);
      }
      
      setError(`${safeT('errors.forecastFailed')}${errorMessage}`);
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
          📊 {safeT('forecast.title')}
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
              🏭 {safeT('forecast.selectAsset')}:
            </label>
            <select
              value={selectedAsset}
              onChange={(e) => setSelectedAsset(e.target.value)}
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
              <option value="">{safeT('forecast.chooseAsset')}</option>
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
              🤖 {safeT('forecast.selectModel')}:
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={!selectedAsset}
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
              <option value="">{safeT('forecast.chooseModel')}</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  v{model.version} - {model.model_type} | MAPE: {model.mape ? `${(model.mape * 100).toFixed(2)}%` : 'N/A'} | {safeT('forecast.trained')}: {new Date(model.created_at).toLocaleDateString()} (ID: {model.id})
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
              ⏱️ {safeT('forecast.forecastHours')}:
            </label>
            <input
              type="number"
              value={forecastHours}
              onChange={(e) => setForecastHours(parseInt(e.target.value))}
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
              📁 {safeT('forecast.dataInputMethod')}:
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
                  onChange={(e) => setDataInputMethod(e.target.value)}
                  style={{ display: 'none' }}
                />
                📤 {safeT('forecast.uploadCSV')}
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
                  onChange={(e) => setDataInputMethod(e.target.value)}
                  style={{ display: 'none' }}
                />
                ☁️ {safeT('forecast.s3Path')}
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
                  {safeT('forecast.uploadHistoricalData')}:
                </label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px dashed #cbd5e0',
                    borderRadius: '12px',
                    background: '#f7fafc',
                    cursor: 'pointer'
                  }}
                />
                {file && (
                  <p style={{ marginTop: '0.5rem', color: '#38a169', fontSize: '0.9rem' }}>
                    ✅ 已选择: {file.name}
                  </p>
                )}
                {!file && savedFileName && (
                  <p style={{ marginTop: '0.5rem', color: '#f56565', fontSize: '0.9rem' }}>
                    💡 上次选择的文件: {savedFileName} (请重新选择)
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
                  {safeT('forecast.s3PathInput')}:
                </label>
                <input
                  type="text"
                  value={s3Path}
                  onChange={(e) => setS3Path(e.target.value)}
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
            disabled={isLoading}
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
            {isLoading ? '⏳' : '🚀'} {isLoading ? safeT('forecast.forecasting') : safeT('forecast.startForecast')}
          </button>
        </div>

        {chartData.datasets.length > 0 && (
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
                      text: safeT('forecast.timestamp') || '时间戳'
                    }
                  },
                  y: {
                    title: {
                      display: true,
                      text: safeT('forecast.energyKwh') || '能耗 (kWh)'
                    }
                  }
                },
                plugins: {
                  legend: {
                    position: 'top',
                  },
                  title: {
                    display: true,
                    text: safeT('forecast.chartTitle') || '能耗预测图表'
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
              data={chartData} 
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default ForecastView;