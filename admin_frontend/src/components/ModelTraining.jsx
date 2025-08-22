import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';
import CustomFileInput from './CustomFileInput';

function ModelTraining() {
  const { t, language } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [modelType, setModelType] = useState('LightGBM');
  const [dataInputMethod, setDataInputMethod] = useState('upload');
  const [s3DataPath, setS3DataPath] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [nEpochs, setNEpochs] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [taskId, setTaskId] = useState(() => {
    return localStorage.getItem('trainingTaskId') || null;
  });
  const [taskStatus, setTaskStatus] = useState(() => {
    return localStorage.getItem('trainingTaskStatus') || null;
  });
  const [taskProgress, setTaskProgress] = useState(() => {
    return localStorage.getItem('trainingTaskProgress') || null;
  });
  const [trainingResult, setTrainingResult] = useState(() => {
    const saved = localStorage.getItem('trainingResult');
    return saved ? JSON.parse(saved) : null;
  });

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
  }, [language]);

  // 任务状态轮询
  useEffect(() => {
    if (!taskId) return;
    
    // 如果有taskId但loading为false，说明是从localStorage恢复的状态，需要重新开始轮询
    if (taskId && !loading && !success && !error) {
      setLoading(true);
    }
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await apiClient(`/admin/tasks/${taskId}/status`);
        const newStatus = status.status;
        const newProgress = status.result?.status || null;
        
        setTaskStatus(newStatus);
        setTaskProgress(newProgress);
        
        // 保存状态到localStorage
        localStorage.setItem('trainingTaskStatus', newStatus);
        localStorage.setItem('trainingTaskProgress', newProgress || '');
        
        if (newStatus === 'SUCCESS') {
          clearInterval(pollInterval);
          setLoading(false);
          setTrainingResult(status.result);
          setSuccess(getText('training.trainingCompleted'));
          
          // 保存结果并清理localStorage
          localStorage.setItem('trainingResult', JSON.stringify(status.result));
          localStorage.removeItem('trainingTaskId');
          localStorage.removeItem('trainingTaskStatus');
          localStorage.removeItem('trainingTaskProgress');
        } else if (newStatus === 'FAILURE') {
          clearInterval(pollInterval);
          setLoading(false);
          setError(getText('training.trainingFailed') + ': ' + (status.result?.error || getText('errors.unexpected')));
          
          // 清理localStorage
          localStorage.removeItem('trainingTaskId');
          localStorage.removeItem('trainingTaskStatus');
          localStorage.removeItem('trainingTaskProgress');
        }
      } catch (err) {
        console.error('Failed to fetch task status:', err);
      }
    }, 2000); // 每2秒轮询一次
    
    return () => clearInterval(pollInterval);
  }, [taskId, getText, loading, success, error]);

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

  // 渲染训练状态
  const renderTrainingStatus = () => {
    if (!taskId || !loading) return null;
    
    const statusIcons = {
      'Initializing training...': '🔧',
      'Downloading training data from S3...': '📥',
      'Training': '🧠',
      'Fitting anomaly detector...': '🔍',
      'Uploading artifacts to S3...': '📤'
    };
    
    // 检查是否包含训练相关的状态
    const getStatusIcon = (progress) => {
      if (!progress) return '🔄';
      for (const [key, icon] of Object.entries(statusIcons)) {
        if (progress.includes(key) || progress.includes(key.toLowerCase())) {
          return icon;
        }
      }
      return '🔄';
    };
    
    return (
      <div style={{
        padding: '1rem',
        background: 'rgba(102, 126, 234, 0.1)',
        borderRadius: '12px',
        border: '2px solid #667eea',
        marginBottom: '1rem'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '1rem',
          color: '#4a5568',
          fontWeight: '500'
        }}>
          <span style={{
            display: 'inline-block',
            animation: 'spin 1s linear infinite'
          }}>
            {getStatusIcon(taskProgress)}
          </span>
          {taskProgress || getText('training.trainingInProgress')}
        </div>
        {taskStatus && (
          <div style={{
            fontSize: '0.8rem',
            color: '#666',
            marginTop: '0.5rem'
          }}>
            {getText('training.status')}: {taskStatus}
          </div>
        )}
      </div>
    );
  };

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    setTaskId(null);
    setTaskStatus(null);
    setTaskProgress(null);
    setTrainingResult(null);

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
      formData.append('asset_id', selectedAsset);
      formData.append('model_type', modelType);
      formData.append('file', selectedFile);
      formData.append('n_epochs', nEpochs.toString());
      formData.append('description', `UI-initiated training for ${selectedAsset} with ${modelType}`);
      options.body = formData;
      url = `/admin/training-jobs-from-csv`;

    } else if (dataInputMethod === 's3') {
      if (!s3DataPath) {
        setError(getText('errors.enterS3Path'));
        setLoading(false);
        return;
      }
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify({
        asset_id: selectedAsset,
        s3_data_path: s3DataPath,
        model_type: modelType,
        n_epochs: nEpochs,
        description: `UI-initiated training for ${selectedAsset} with ${modelType}`
      });
      url = `/admin/training-jobs`;
    }

    try {
      const result = await apiClient(url, options);
      setTaskId(result.task_id);
      
      // 保存taskId到localStorage
      localStorage.setItem('trainingTaskId', result.task_id);
      // 清理之前的结果
      localStorage.removeItem('trainingResult');
      
      // 不在这里设置loading为false，让状态轮询来控制
    } catch (err) {
      console.error("Training error:", err);
      setError(getText('errors.trainingFailed') + (err.message || getText('errors.unexpected')));
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      {/* 添加CSS动画 */}
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
      
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
              disabled={loading}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: loading ? '#f7fafc' : 'white',
                transition: 'all 0.3s ease',
                cursor: loading ? 'not-allowed' : 'pointer'
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
              disabled={loading}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: loading ? '#f7fafc' : 'white',
                transition: 'all 0.3s ease',
                cursor: loading ? 'not-allowed' : 'pointer'
              }}
            >
              <option value="LightGBM">LightGBM</option>
              <option value="TiDE">TiDE (Time-series Dense Encoder)</option>
              <option value="LSTM">LSTM (Long Short-Term Memory)</option>
              <option value="TFT">TFT (Temporal Fusion Transformer)</option>
              <option value="TFT (No Past Covariates)">TFT (No Past Covariates)</option>
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
              🔢 {getText('training.epochs')}:
            </label>
            <input
              type="number"
              value={nEpochs}
              onChange={e => setNEpochs(parseInt(e.target.value, 10))}
              min="1"
              max="200"
              disabled={loading}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '1rem',
                background: loading ? '#f7fafc' : 'white',
                transition: 'all 0.3s ease',
                cursor: loading ? 'not-allowed' : 'text'
              }}
            />
            <small style={{
              color: '#666',
              fontSize: '0.8rem',
              marginTop: '0.25rem',
              display: 'block'
            }}>
              {getText('training.epochsHint')}
            </small>
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
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s ease',
                border: '2px solid',
                borderColor: dataInputMethod === 'upload' ? '#667eea' : '#e2e8f0',
                opacity: loading ? 0.6 : 1
              }}>
                <input
                  type="radio"
                  value="upload"
                  checked={dataInputMethod === 'upload'}
                  onChange={() => !loading && setDataInputMethod('upload')}
                  disabled={loading}
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
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s ease',
                border: '2px solid',
                borderColor: dataInputMethod === 's3' ? '#667eea' : '#e2e8f0',
                opacity: loading ? 0.6 : 1
              }}>
                <input
                  type="radio"
                  value="s3"
                  checked={dataInputMethod === 's3'}
                  onChange={() => !loading && setDataInputMethod('s3')}
                  disabled={loading}
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
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px dashed #cbd5e0',
                    borderRadius: '12px',
                    background: loading ? '#f7fafc' : '#f7fafc',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.6 : 1
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
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px solid #e2e8f0',
                    borderRadius: '12px',
                    fontSize: '1rem',
                    background: loading ? '#f7fafc' : 'white',
                    cursor: loading ? 'not-allowed' : 'text'
                  }}
                />
              </div>
            )}
          </div>

          {/* 训练状态显示 */}
          {renderTrainingStatus()}

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
              {trainingResult && trainingResult.mape && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                  📊 MAPE: {trainingResult.mape.toFixed(2)}%
                </div>
              )}
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
            {loading ? '⏳' : '🚀'} {loading ? getText('training.trainingInProgress') : getText('training.startTraining')}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ModelTraining;