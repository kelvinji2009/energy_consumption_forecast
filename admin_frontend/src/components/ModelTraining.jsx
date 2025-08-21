import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function ModelTraining({ activeTask, setActiveTask }) {
    const { t } = useLanguage();
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [s3DataPath, setS3DataPath] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [dataInputMethod, setDataInputMethod] = useState('upload');
    const [nEpochs, setNEpochs] = useState(20);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState('LightGBM');

    // 安全翻译函数
    const safeT = (key) => {
        if (typeof t === 'function') {
            return t(key);
        }
        // 提供默认的中文文本
        const defaultTexts = {
            'training.algorithms.LightGBM.label': 'LightGBM',
            'training.algorithms.LightGBM.desc': '轻量级梯度提升机，适合快速训练和高精度预测',
            'training.algorithms.TiDE.label': 'TiDE',
            'training.algorithms.TiDE.desc': '时间序列密集编码器，专为长期预测设计',
            'training.algorithms.LSTM.label': 'LSTM',
            'training.algorithms.LSTM.desc': '长短期记忆网络，擅长处理序列数据',
            'training.algorithms.TFT.label': 'TFT',
            'training.algorithms.TFT.desc': '时间融合变换器，支持多变量时间序列预测',
            'training.algorithms.TFT (No Past Covariates).label': 'TFT (无历史协变量)',
            'training.algorithms.TFT (No Past Covariates).desc': '简化版TFT，不使用历史协变量',
            'training.title': '模型训练',
            'training.selectAsset': '选择资产',
            'training.selectAlgorithm': '选择算法',
            'training.dataInputMethod': '数据输入方式',
            'training.uploadCsv': '上传CSV文件',
            'training.s3Path': 'S3路径',
            'training.trainingDataFile': '训练数据文件',
            'training.fileSelected': '已选择文件',
            'training.s3DataPath': 'S3数据路径',
            'training.s3Placeholder': '例如: s3://bucket/path/to/data.csv',
            'training.epochs': '训练轮数',
            'training.epochsHint': '建议值：LightGBM 10-50轮，深度学习模型 20-100轮',
            'training.starting': '正在启动...',
            'training.taskRunning': '任务运行中...',
            'training.noAssets': '没有可用资产',
            'training.startTraining': '开始训练',
            'training.errors.failedToLoadAssets': '加载资产失败',
            'training.errors.noFile': '请选择文件',
            'training.errors.noS3Path': '请输入S3路径',
            'training.errors.failedToStart': '启动训练失败'
        };
        return defaultTexts[key] || key;
    };

    const algorithms = [
        { value: 'LightGBM', label: safeT('training.algorithms.LightGBM.label'), description: safeT('training.algorithms.LightGBM.desc') },
        { value: 'TiDE', label: safeT('training.algorithms.TiDE.label'), description: safeT('training.algorithms.TiDE.desc') },
        { value: 'LSTM', label: safeT('training.algorithms.LSTM.label'), description: safeT('training.algorithms.LSTM.desc') },
        { value: 'TFT', label: safeT('training.algorithms.TFT.label'), description: safeT('training.algorithms.TFT.desc') },
        { value: 'TFT (No Past Covariates)', label: safeT('training.algorithms.TFT (No Past Covariates).label'), description: safeT('training.algorithms.TFT (No Past Covariates).desc') }
    ];

    useEffect(() => {
        const fetchAssets = async () => {
            try {
                const data = await apiClient('/admin/assets');
                setAssets(data);
                if (data.length > 0) {
                    setSelectedAsset(data[0].id);
                }
            } catch (err) {
                console.error("Failed to fetch assets:", err);
                setError(safeT('training.errors.failedToLoadAssets'));
            }
        };
        fetchAssets();
    }, [safeT]);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError(null);
        setIsLoading(true);

        try {
            let data;
            if (dataInputMethod === 'upload') {
                if (!selectedFile) {
                    throw new Error(safeT('training.errors.noFile'));
                }
                const formData = new FormData();
                formData.append('asset_id', selectedAsset);
                formData.append('model_type', selectedAlgorithm);
                formData.append('n_epochs', nEpochs);
                formData.append('description', `UI训练任务：${selectedAsset} - ${selectedAlgorithm}`);
                formData.append('file', selectedFile);
                
                data = await apiClient('/admin/training-jobs-from-csv', {
                    method: 'POST',
                    body: formData,
                });

            } else {
                if (!s3DataPath) {
                    throw new Error(safeT('training.errors.noS3Path'));
                }
                const jobRequest = {
                    asset_id: selectedAsset,
                    s3_data_path: s3DataPath,
                    model_type: selectedAlgorithm,
                    description: `UI-S3训练任务：${selectedAsset} - ${selectedAlgorithm}`,
                    n_epochs: nEpochs,
                };
                data = await apiClient('/admin/training-jobs', {
                    method: 'POST',
                    body: JSON.stringify(jobRequest),
                });
            }
            
            setActiveTask({ id: data.task_id, status: data.status });
            setS3DataPath('');
            setSelectedFile(null);

        } catch (err) {
            console.error('Training job error:', err);
            let errorMessage = '发生了意外错误。';
            
            if (err.message) {
                errorMessage = err.message;
            } else if (typeof err === 'string') {
                errorMessage = err;
            } else if (err.detail) {
                errorMessage = err.detail;
            } else {
                errorMessage = JSON.stringify(err);
            }
            
            setError(`${safeT('training.errors.failedToStart')}：${errorMessage}`);
        } finally {
            setIsLoading(false);
        }
    };

    const selectedAlgorithmInfo = algorithms.find(algo => algo.value === selectedAlgorithm);

    return (
        <div style={{ padding: '0', fontFamily: 'inherit' }}>
            <h2 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {safeT('training.title')}
            </h2>
            
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {/* 资产选择 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        {safeT('training.selectAsset')}
                    </label>
                    <select 
                        value={selectedAsset} 
                        onChange={e => setSelectedAsset(e.target.value)} 
                        required 
                        disabled={isLoading || activeTask}
                        style={{ width: '100%', padding: '0.75rem 1rem' }}
                    >
                        {assets.map(asset => (
                            <option key={asset.id} value={asset.id}>
                                {asset.name} ({asset.id})
                            </option>
                        ))}
                    </select>
                </div>

                {/* 算法选择 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        {safeT('training.selectAlgorithm')}
                    </label>
                    <select 
                        value={selectedAlgorithm} 
                        onChange={e => setSelectedAlgorithm(e.target.value)} 
                        required 
                        disabled={isLoading || activeTask}
                        style={{ width: '100%', padding: '0.75rem 1rem' }}
                    >
                        {algorithms.map(algo => (
                            <option key={algo.value} value={algo.value}>
                                {algo.label}
                            </option>
                        ))}
                    </select>
                    {selectedAlgorithmInfo && (
                        <small style={{ 
                            color: '#718096', 
                            fontSize: '0.85rem', 
                            marginTop: '0.5rem', 
                            display: 'block',
                            padding: '0.5rem',
                            background: 'rgba(102, 126, 234, 0.1)',
                            borderRadius: '8px',
                            border: '1px solid rgba(102, 126, 234, 0.2)'
                        }}>
                            💡 {selectedAlgorithmInfo.description}
                        </small>
                    )}
                </div>

                {/* 数据输入方式选择 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                        {safeT('training.dataInputMethod')}
                    </label>
                    <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                        <label style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '0.5rem',
                            padding: '0.75rem 1.5rem',
                            background: dataInputMethod === 'upload' ? 'rgba(102, 126, 234, 0.1)' : 'rgba(255, 255, 255, 0.7)',
                            border: `2px solid ${dataInputMethod === 'upload' ? '#667eea' : 'rgba(255, 255, 255, 0.3)'}`,
                            borderRadius: '10px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease'
                        }}>
                            <input 
                                type="radio" 
                                value="upload" 
                                checked={dataInputMethod === 'upload'} 
                                onChange={() => setDataInputMethod('upload')} 
                                disabled={isLoading || activeTask}
                                style={{ width: 'auto', margin: 0 }}
                            />
                            {safeT('training.uploadCsv')}
                        </label>
                        <label style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '0.5rem',
                            padding: '0.75rem 1.5rem',
                            background: dataInputMethod === 's3' ? 'rgba(102, 126, 234, 0.1)' : 'rgba(255, 255, 255, 0.7)',
                            border: `2px solid ${dataInputMethod === 's3' ? '#667eea' : 'rgba(255, 255, 255, 0.3)'}`,
                            borderRadius: '10px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease'
                        }}>
                            <input 
                                type="radio" 
                                value="s3" 
                                checked={dataInputMethod === 's3'} 
                                onChange={() => setDataInputMethod('s3')} 
                                disabled={isLoading || activeTask}
                                style={{ width: 'auto', margin: 0 }}
                            />
                            {safeT('training.s3Path')}
                        </label>
                    </div>
                </div>

                {/* 条件数据输入 */}
                {dataInputMethod === 'upload' ? (
                    <div>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                            {safeT('training.trainingDataFile')}
                        </label>
                        <input 
                            type="file" 
                            accept=".csv" 
                            onChange={handleFileChange} 
                            required 
                            disabled={isLoading || activeTask}
                            style={{ 
                                width: '100%', 
                                padding: '1rem',
                                border: '2px dashed rgba(102, 126, 234, 0.3)',
                                background: 'rgba(102, 126, 234, 0.05)',
                                borderRadius: '10px'
                            }}
                        />
                        {selectedFile && (
                            <div style={{ 
                                marginTop: '0.5rem',
                                padding: '0.5rem 1rem',
                                background: 'rgba(72, 187, 120, 0.1)',
                                border: '1px solid rgba(72, 187, 120, 0.2)',
                                borderRadius: '8px',
                                color: '#48bb78',
                                fontSize: '0.9rem'
                            }}>
                                {safeT('training.fileSelected')}: {selectedFile.name}
                            </div>
                        )}
                    </div>
                ) : (
                    <div>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                            {safeT('training.s3DataPath')}
                        </label>
                        <input 
                            type="text" 
                            value={s3DataPath} 
                            onChange={e => setS3DataPath(e.target.value)} 
                            placeholder={safeT('training.s3Placeholder')}
                            required 
                            disabled={isLoading || activeTask}
                            style={{ width: '100%', padding: '0.75rem 1rem' }}
                        />
                    </div>
                )}

                {/* 训练参数 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        {safeT('training.epochs')}
                    </label>
                    <input 
                        type="number" 
                        value={nEpochs} 
                        onChange={e => setNEpochs(parseInt(e.target.value, 10))} 
                        min="1" 
                        max="200" 
                        required 
                        disabled={isLoading || activeTask}
                        style={{ width: '100%', padding: '0.75rem 1rem' }}
                    />
                    <small style={{ 
                        color: '#718096', 
                        fontSize: '0.85rem', 
                        marginTop: '0.5rem', 
                        display: 'block',
                        padding: '0.5rem',
                        background: 'rgba(255, 193, 7, 0.1)',
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 193, 7, 0.2)'
                    }}>
                        {safeT('training.epochsHint')}
                    </small>
                </div>

                {/* 提交按钮 */}
                <button 
                    type="submit" 
                    disabled={isLoading || activeTask || assets.length === 0}
                    style={{ 
                        padding: '1rem 2rem',
                        fontSize: '1rem',
                        fontWeight: '600',
                        cursor: isLoading || activeTask || assets.length === 0 ? 'not-allowed' : 'cursor',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem'
                    }}
                >
                    {isLoading ? (
                        <>
                            <span className="pulse">⏳</span> {safeT('training.starting')}
                        </>
                    ) : activeTask ? (
                        <>
                            <span className="pulse">🔄</span> {safeT('training.taskRunning')}
                        </>
                    ) : assets.length === 0 ? (
                        <>{safeT('training.noAssets')}</>
                    ) : (
                        <>{safeT('training.startTraining')}</>
                    )}
                </button>
            </form>
            
            {error && (
                <div className="error-message" style={{ marginTop: '1.5rem' }}>
                    ❌ {error}
                </div>
            )}
        </div>
    );
}

export default ModelTraining;