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

function AnomalyDetectionView() {
  const { t, language } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [dataInputMethod, setDataInputMethod] = useState('upload');
  const [s3DataPath, setS3DataPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState({ datasets: [] });
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [sensitivity, setSensitivity] = useState(0.95);
  const chartRef = useRef(null);

  // 改进的翻译函数，确保能够正确获取当前语言的翻译
  const getText = (key) => {
    if (!t || typeof t !== 'object') {
      return key;
    }
    
    // 支持嵌套键，如 'forecast.title'
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

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const data = await apiClient('/admin/assets');
        setAssets(data);
        if (data.length > 0) {
          setSelectedAsset(prev => prev || data[0].id);
        }
      } catch (err) {
        console.error("Failed to fetch assets:", err);
        setError(getText('errors.fetchAssets'));
      }
    };
    fetchAssets();
  }, [language]); // 添加language依赖，确保语言切换时重新获取资产列表

  useEffect(() => {
    if (selectedAsset) {
      setLoading(true);
      setError(null);
      setModels([]);
      setSelectedModelId('');
      const fetchModels = async () => {
        try {
          const data = await apiClient(`/admin/models?asset_id=${selectedAsset}`);
          const modelsWithDetectors = data
            .filter(m => m.status === 'COMPLETED' && m.detector_path);
          const sortedModels = modelsWithDetectors.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          
          setModels(sortedModels);
          if (sortedModels.length > 0) {
            setSelectedModelId(sortedModels[0].id);
          } else {
            setError(getText('anomaly.noDetectors'));
          }
        } catch (err) {
          console.error("Failed to fetch models:", err);
          setError(getText('errors.fetchModels'));
        } finally {
          setLoading(false);
        }
      };
      fetchModels();
    }
  }, [selectedAsset, language]); // 添加language依赖

  const handleFileChange = (file) => {
    if (file && file.type === "text/csv") {
      setSelectedFile(file);
      setFileName(file.name);
      setError(null);
    } else {
      setSelectedFile(null);
      setFileName('');
      setError(getText('errors.invalidFile'));
    }
  };

  const handleDetect = async () => {
    setLoading(true);
    setError(null);
    setChartData({ datasets: [] });

    if (!selectedAsset || !selectedModelId) {
      setError(getText('errors.selectAssetModel'));
      setLoading(false);
      return;
    }

    let url = '';
    const options = { method: 'POST' };

    if (dataInputMethod === 'upload') {
      if (!selectedFile) {
        setError(getText('errors.selectFile'));
        setLoading(false);
        return;
      }
      const formData = new FormData();
      formData.append('model_id', selectedModelId);
      formData.append('sensitivity', sensitivity.toString());
      formData.append('file', selectedFile);
      options.body = formData;
      url = `/assets/${selectedAsset}/detect_anomalies_from_csv`;

    } else if (dataInputMethod === 's3') {
      if (!s3DataPath) {
        setError(getText('errors.enterS3Path'));
        setLoading(false);
        return;
      }
      const params = new URLSearchParams({
        s3_data_path: s3DataPath,
        model_id: selectedModelId,
        sensitivity: sensitivity.toString(),
      });
      url = `/assets/${selectedAsset}/detect_anomalies_from_s3?${params.toString()}`;
    }

    try {
      const result = await apiClient(url, options);

      const historicalData = result.historical_data
        .filter(d => d.timestamp && d.value != null)
        .map(d => ({ x: new Date(d.timestamp), y: d.value }));

      const anomalyPoints = result.anomalies.map(a => ({
        x: new Date(a.timestamp),
        y: a.value
      }));

      setChartData({
        datasets: [
          {
            label: getText('anomaly.historicalEnergy'),
            data: historicalData,
            borderColor: '#8884d8',
            backgroundColor: 'rgba(136, 132, 216, 0.5)',
            pointRadius: 1,
            type: 'line',
          },
          {
            label: getText('anomaly.anomalies'),
            data: anomalyPoints,
            backgroundColor: 'red',
            pointRadius: 5,
            type: 'scatter',
          }
        ]
      });

    } catch (err) {
      console.error("Detection error:", err);
      setError(getText('errors.detectionFailed') + (err.message || getText('errors.unexpected')));
    } finally {
      setLoading(false);
    }
  };

  const chartOptions = {
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
          text: getText('anomaly.timestamp')
        }
      },
      y: {
        title: {
          display: true,
          text: getText('anomaly.energyKwh')
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: getText('anomaly.chartTitle')
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
          🚨 {getText('anomaly.title')}
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
              🏭 {getText('anomaly.selectAsset')}:
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
              🤖 {getText('anomaly.selectModel')}:
            </label>
            <select
              value={selectedModelId}
              onChange={e => setSelectedModelId(e.target.value)}
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
              {models.length === 0 ? (
                <option value="">{getText('anomaly.noModels')}</option>
              ) : (
                models.map(model => (
                  <option key={model.id} value={model.id}>
                    v{model.model_version} - {model.model_type} | MAPE: {model.metrics?.mape?.toFixed(2) ?? 'N/A'}% | {getText('anomaly.trained')}: {new Date(model.created_at).toLocaleDateString()} (ID: {model.id})
                  </option>
                ))
              )}
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
              🎯 {getText('anomaly.sensitivity')}:
            </label>
            <div style={{ marginBottom: '1rem' }}>
              <input
                type="range"
                min="0.80"
                max="0.99"
                step="0.01"
                value={sensitivity}
                onChange={e => {
                  const value = parseFloat(e.target.value);
                  // 确保值在有效范围内
                  if (value >= 0.80 && value <= 0.99) {
                    setSensitivity(value);
                  }
                }}
                style={{
                  width: '100%',
                  height: '6px',
                  borderRadius: '3px',
                  background: '#e2e8f0',
                  outline: 'none',
                  marginBottom: '0.5rem'
                }}
              />
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.85rem',
                color: '#718096'
              }}>
                <span>🔍 {getText('anomaly.moreAnomalies')}</span>
                <span style={{
                  background: '#667eea',
                  color: 'white',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '15px',
                  fontWeight: '600'
                }}>
                  {(sensitivity * 100).toFixed(0)}%
                </span>
                <span>🎯 {getText('anomaly.fewerAnomalies')}</span>
              </div>
              <div style={{
                fontSize: '0.8rem',
                color: '#a0aec0',
                textAlign: 'center',
                marginTop: '0.5rem'
              }}>
                {getText('anomaly.sensitivityDesc')}
              </div>
            </div>
          </div>

          <div>
            <label style={{
              display: 'block',
              marginBottom: '1rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              📁 {getText('anomaly.dataInputMethod')}:
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
                📤 {getText('anomaly.uploadCSV')}
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
                ☁️ {getText('anomaly.s3Path')}
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
                  {getText('anomaly.uploadHistoricalData')}:
                </label>
                <CustomFileInput
                  accept=".csv"
                  onFileChange={handleFileChange}
                  selectedFile={selectedFile}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px dashed #cbd5e0',
                    borderRadius: '12px',
                    background: '#f7fafc',
                    cursor: 'pointer'
                  }}
                />
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
                  {getText('anomaly.s3PathInput')}:
                </label>
                <input
                  type="text"
                  value={s3DataPath}
                  onChange={e => setS3DataPath(e.target.value)}
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
            onClick={handleDetect}
            disabled={loading || !assets.length || !models.length}
            style={{
              padding: '1rem 2rem',
              background: loading ? '#a0aec0' : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '25px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(240, 147, 251, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            {loading ? '⏳' : '🚨'} {loading ? getText('anomaly.detecting') : getText('anomaly.startDetection')}
          </button>
        </div>

        <div style={{ position: 'relative', width: '100%', height: '400px', marginTop: '2rem' }}>
          <Line ref={chartRef} options={chartOptions} data={chartData} />
        </div>
      </div>
    </div>
  );
}

export default AnomalyDetectionView;