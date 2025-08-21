import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';

function ModelTraining({ activeTask, setActiveTask }) {
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [s3DataPath, setS3DataPath] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [dataInputMethod, setDataInputMethod] = useState('upload');
    const [nEpochs, setNEpochs] = useState(20);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState('LightGBM');

    const algorithms = [
        { value: 'LightGBM', label: '🚀 LightGBM', description: '快速梯度提升，适合大数据集' },
        { value: 'TiDE', label: '🌊 TiDE', description: '时间序列密集编码器，高效准确' },
        { value: 'LSTM', label: '🧠 LSTM', description: '长短期记忆网络，处理序列依赖' },
        { value: 'TFT', label: '🎯 TFT', description: '时间融合变换器，最高精度' },
        { value: 'TFT (No Past Covariates)', label: '⚡ TFT (简化版)', description: 'TFT 无历史协变量版本' }
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
                setError("无法加载资产列表。请检查 API 服务器是否运行正常以及 API 密钥是否正确。");
            }
        };
        fetchAssets();
    }, []);

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
                    throw new Error("请选择要上传的 CSV 文件。");
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
                    throw new Error("请提供 S3 数据路径。");
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
            
            setError(`训练任务启动失败：${errorMessage}`);
        } finally {
            setIsLoading(false);
        }
    };

    const selectedAlgorithmInfo = algorithms.find(algo => algo.value === selectedAlgorithm);

    return (
        <div style={{ padding: '0', fontFamily: 'inherit' }}>
            <h2 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                🎯 开始新的模型训练
            </h2>
            
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {/* 资产选择 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        🏭 选择资产
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
                        🤖 选择算法
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
                        📁 数据输入方式
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
                            📤 上传 CSV 文件
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
                            ☁️ S3 路径
                        </label>
                    </div>
                </div>

                {/* 条件数据输入 */}
                {dataInputMethod === 'upload' ? (
                    <div>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                            📊 训练数据文件
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
                                ✅ 已选择文件: {selectedFile.name}
                            </div>
                        )}
                    </div>
                ) : (
                    <div>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                            ☁️ S3 数据路径 (Key)
                        </label>
                        <input 
                            type="text" 
                            value={s3DataPath} 
                            onChange={e => setS3DataPath(e.target.value)} 
                            placeholder="例如: training-data/data.csv" 
                            required 
                            disabled={isLoading || activeTask}
                            style={{ width: '100%', padding: '0.75rem 1rem' }}
                        />
                    </div>
                )}

                {/* 训练参数 */}
                <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        ⚙️ 训练轮数 (Epochs)
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
                        💡 神经网络模型 (TiDE, LSTM, TFT) 推荐 20-100 轮。LightGBM 不使用此参数。
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
                        cursor: isLoading || activeTask || assets.length === 0 ? 'not-allowed' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem'
                    }}
                >
                    {isLoading ? (
                        <>
                            <span className="pulse">⏳</span> 正在启动训练任务...
                        </>
                    ) : activeTask ? (
                        <>
                            <span className="pulse">🔄</span> 训练任务进行中
                        </>
                    ) : assets.length === 0 ? (
                        <>❌ 无可用资产</>
                    ) : (
                        <>🚀 开始训练任务</>
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