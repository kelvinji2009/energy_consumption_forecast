import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function ForecastView() {
  const { t } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [forecastHours, setForecastHours] = useState(168);
  const [dataInputMethod, setDataInputMethod] = useState('upload');
  const [file, setFile] = useState(null);
  const [s3Path, setS3Path] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [forecastResult, setForecastResult] = useState(null);

  useEffect(() => {
    fetchAssets();
  }, []);

  useEffect(() => {
    if (selectedAsset) {
      fetchModels(selectedAsset);
    }
  }, [selectedAsset]);

  const fetchAssets = async () => {
    try {
      const data = await apiClient('/admin/assets');
      setAssets(data);
    } catch (err) {
      setError(t.errors.fetchAssets);
    }
  };

  const fetchModels = async (assetId) => {
    try {
      const data = await apiClient(`/admin/models?asset_id=${assetId}`);
      setModels(data.filter(model => model.status === 'COMPLETED'));
    } catch (err) {
      setError(t.errors.fetchModels);
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleForecast = async () => {
    if (!selectedAsset || !selectedModel) {
      setError(t.errors.selectAssetModel);
      return;
    }

    if (dataInputMethod === 'upload' && !file) {
      setError(t.errors.selectFile);
      return;
    }

    if (dataInputMethod === 's3' && !s3Path) {
      setError(t.errors.enterS3Path);
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('model_id', selectedModel);
      formData.append('forecast_horizon', forecastHours);

      if (dataInputMethod === 'upload') {
        formData.append('file', file);
      } else {
        formData.append('s3_path', s3Path);
      }

      const result = await apiClient(`/assets/${selectedAsset}/predict`, {
        method: 'POST',
        body: formData,
        headers: {},
      });

      setForecastResult(result);
    } catch (err) {
      setError(t.errors.forecastFailed + (err.message || t.errors.unexpected));
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
          📊 {t.forecast.title}
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
              🏭 {t.forecast.selectAsset}:
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
              <option value="">{t.forecast.chooseAsset}</option>
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
              🤖 {t.forecast.selectModel}:
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
              <option value="">{t.forecast.chooseModel}</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  v{model.version} - {model.model_type} | MAPE: {model.mape ? `${(model.mape * 100).toFixed(2)}%` : 'N/A'} | {t.forecast.trained}: {new Date(model.created_at).toLocaleDateString()} (ID: {model.id})
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
              ⏱️ {t.forecast.forecastHours}:
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
              📁 {t.forecast.dataInputMethod}:
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
                📤 {t.forecast.uploadCSV}
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
                ☁️ {t.forecast.s3Path}
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
                  {t.forecast.uploadHistoricalData}:
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
                    {file.name}
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
                  {t.forecast.s3PathInput}:
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
            {isLoading ? '⏳' : '🚀'} {isLoading ? t.forecast.forecasting : t.forecast.startForecast}
          </button>
        </div>

        {forecastResult && (
          <div style={{
            marginTop: '2rem',
            padding: '1.5rem',
            background: 'rgba(236, 253, 245, 0.9)',
            borderRadius: '12px',
            border: '1px solid #9ae6b4'
          }}>
            <h3 style={{ color: '#38a169', marginBottom: '1rem' }}>
              {t.forecast.results}
            </h3>
            <pre style={{
              background: 'white',
              padding: '1rem',
              borderRadius: '8px',
              overflow: 'auto',
              fontSize: '0.9rem'
            }}>
              {JSON.stringify(forecastResult, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default ForecastView;