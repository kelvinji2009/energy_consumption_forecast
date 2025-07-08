import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';

function ModelTraining({ activeTask, setActiveTask }) {
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [s3DataPath, setS3DataPath] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [dataInputMethod, setDataInputMethod] = useState('upload'); // 'upload' or 's3'
    const [nEpochs, setNEpochs] = useState(20);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState('LightGBM');

    const algorithms = ['LightGBM', 'TiDE', 'LSTM', 'TFT', 'TFT (No Past Covariates)'];

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
                setError("Could not load assets. Is the API server running and the API key correct?");
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
                    throw new Error("Please select a CSV file to upload.");
                }
                const formData = new FormData();
                formData.append('asset_id', selectedAsset);
                formData.append('model_type', selectedAlgorithm);
                formData.append('n_epochs', nEpochs);
                formData.append('description', `UI-upload training for ${selectedAsset} with ${selectedAlgorithm}`);
                formData.append('file', selectedFile);
                
                data = await apiClient('/admin/training-jobs-from-csv', {
                    method: 'POST',
                    body: formData,
                    // Let browser set Content-Type for multipart/form-data
                });

            } else { // 's3'
                if (!s3DataPath) {
                    throw new Error("Please provide an S3 data path.");
                }
                const jobRequest = {
                    asset_id: selectedAsset,
                    s3_data_path: s3DataPath,
                    model_type: selectedAlgorithm,
                    description: `UI-S3 training for ${selectedAsset} with ${selectedAlgorithm}`,
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
            setError(`Failed to start training job: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
            <h2>Start New Model Training</h2>
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '500px' }}>
                {/* Asset and Algorithm Selection */}
                <div>
                    <label>Asset</label>
                    <select value={selectedAsset} onChange={e => setSelectedAsset(e.target.value)} required disabled={isLoading || activeTask} style={{ width: '100%', padding: '0.5rem' }}>
                        {assets.map(asset => <option key={asset.id} value={asset.id}>{asset.name} ({asset.id})</option>)}
                    </select>
                </div>
                <div>
                    <label>Algorithm</label>
                    <select value={selectedAlgorithm} onChange={e => setSelectedAlgorithm(e.target.value)} required disabled={isLoading || activeTask} style={{ width: '100%', padding: '0.5rem' }}>
                        {algorithms.map(algo => <option key={algo} value={algo}>{algo}</option>)}
                    </select>
                </div>

                {/* Data Input Method Selection */}
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <label><input type="radio" value="upload" checked={dataInputMethod === 'upload'} onChange={() => setDataInputMethod('upload')} disabled={isLoading || activeTask} /> Upload CSV</label>
                    <label><input type="radio" value="s3" checked={dataInputMethod === 's3'} onChange={() => setDataInputMethod('s3')} disabled={isLoading || activeTask} /> S3 Path</label>
                </div>

                {/* Conditional Data Input */}
                {dataInputMethod === 'upload' ? (
                    <div>
                        <label>Training Data File</label>
                        <input type="file" accept=".csv" onChange={handleFileChange} required style={{ width: '100%', padding: '0.5rem' }} disabled={isLoading || activeTask} />
                        {selectedFile && <small>{selectedFile.name}</small>}
                    </div>
                ) : (
                    <div>
                        <label>S3 Data Path (Key)</label>
                        <input type="text" value={s3DataPath} onChange={e => setS3DataPath(e.target.value)} placeholder="e.g., training-data/data.csv" required style={{ width: '100%', padding: '0.5rem' }} disabled={isLoading || activeTask} />
                    </div>
                )}

                {/* Epochs and Submit Button */}
                <div>
                    <label>Number of Epochs</label>
                    <input type="number" value={nEpochs} onChange={e => setNEpochs(parseInt(e.target.value, 10))} min="1" max="200" required style={{ width: '100%', padding: '0.5rem' }} disabled={isLoading || activeTask} />
                    <small style={{ color: '#666', fontSize: '0.8rem', marginTop: '0.25rem' }}>
                        Recommended for neural networks (TiDE, LSTM, TFT): 20-100. LightGBM does not use epochs.
                    </small>
                </div>
                <button type="submit" disabled={isLoading || activeTask || assets.length === 0} style={{ padding: '0.75rem', cursor: 'pointer' }}>
                    {isLoading ? 'Starting Job...' : (activeTask ? 'A Task is Running' : 'Start Training Job')}
                </button>
            </form>
            {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}
        </div>
    );
}

export default ModelTraining;
