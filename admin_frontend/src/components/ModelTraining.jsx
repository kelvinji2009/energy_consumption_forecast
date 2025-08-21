import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';
import CustomFileInput from './CustomFileInput';

function ModelTraining() {
  const { t, language } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [modelType, setModelType] = useState('tft');
  const [dataInputMethod, setDataInputMethod] = useState('upload');
  const [s3DataPath, setS3DataPath] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // 改进的翻译函数，确保能够正确获取当前语言的翻译
  const getText = (key) => {
    if (!t || typeof t !== 'object') {
      return key;
    }
    
    // 支持嵌套键，如 'training.title'
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

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    if (!selectedAsset) {
      setError(getText('errors.selectAsset'));
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
      formData.append('model_type', modelType);
      formData.append('file', selectedFile);
      options.body = formData;
      url = `/assets/${selectedAsset}/train_from_csv`;

    } else if (dataInputMethod === 's3') {
      if (!s3DataPath) {
        setError(getText('errors.enterS3Path'));
        setLoading(false);
        return;
      }
      const params = new URLSearchParams({
        s3_data_path: s3DataPath,
        model_type: modelType,
      });
      url = `/assets/${selectedAsset}/train_from_s3?${params.toString()}`;
    }

    try {
      const result = await apiClient(url, options);
      setSuccess(getText('training.trainingStarted') + ` ${getText('training.taskId')}: ${result.task_id}`);
    } catch (err) {
      console.error("Training error:", err);
      setError(getText('errors.trainingFailed') + (err.message || getText('errors.unexpected')));
    } finally {
      setLoading(false);
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
          🤖 {getText('training.title')}
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
              🏭 {getText('training.selectAsset')}:
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
              🧠 {getText('training.selectModelType')}:
            </label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value)}
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
              <option value="tft">TFT (Temporal Fusion Transformer)</option>
              <option value="lstm">LSTM (Long Short-Term Memory)</option>
              <option value="lgbm">LightGBM</option>
              <option value="tide">TiDE (Time-series Dense Encoder)</option>
            </select>
          </div>

          <div>
            <label style={{
              display: 'block',
              marginBottom: '1rem',
              fontWeight: '500',
              color: '#4a5568',
              fontSize: '0.9rem'
            }}>
              📁 {getText('training.dataInputMethod')}:
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
                📤 {getText('training.uploadCSV')}
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
                ☁️ {getText('training.s3Path')}
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
                  {getText('training.uploadTrainingData')}:
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
                  {getText('training.s3PathInput')}:
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

          {success && (
            <div style={{
              padding: '1rem',
              background: 'rgba(198, 246, 213, 0.9)',
              color: '#2f855a',
              borderRadius: '12px',
              border: '1px solid #9ae6b4'
            }}>
              {success}
            </div>
          )}

          <button
            onClick={handleTrain}
            disabled={loading || !assets.length}
            style={{
              padding: '1rem 2rem',
              background: loading ? '#a0aec0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '25px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            {loading ? '⏳' : '🚀'} {loading ? getText('training.training') : getText('training.startTraining')}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ModelTraining;