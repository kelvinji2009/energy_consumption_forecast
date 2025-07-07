import React, { useState, useEffect } from 'react';

function ModelTraining() {
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [s3DataPath, setS3DataPath] = useState('');
    const [nEpochs, setNEpochs] = useState(20); // New state for n_epochs
    const [message, setMessage] = useState(null);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // Fetch assets for the dropdown
    useEffect(() => {
        fetch('/api/admin/assets')
            .then(response => response.json())
            .then(data => {
                setAssets(data);
                if (data.length > 0) {
                    setSelectedAsset(data[0].id);
                }
            })
            .catch(err => {
                console.error("Failed to fetch assets:", err);
                setError("Could not load assets. Is the API server running?");
            });
    }, []);

    const [selectedAlgorithm, setSelectedAlgorithm] = useState('LightGBM'); // New state for algorithm selection

    const algorithms = ['LightGBM', 'TiDE', 'LSTM', 'TFT', 'TFT (No Past Covariates)']; // Available algorithms

    const handleSubmit = (event) => {
        event.preventDefault();
        setMessage(null);
        setError(null);
        setIsLoading(true);

        const jobRequest = {
            asset_id: selectedAsset,
            s3_data_path: s3DataPath,
            model_type: selectedAlgorithm,
            description: `UI-initiated training for ${selectedAsset} with ${selectedAlgorithm}`,
            n_epochs: nEpochs // Include n_epochs in the request
        };

        fetch('/api/admin/training-jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jobRequest),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail || 'An unknown error occurred.') });
            }
            return response.json();
        })
        .then(data => {
            setMessage(`Training job started successfully! Model ID: ${data.model_id}, Task ID: ${data.task_id}`);
            setS3DataPath(''); // Clear input on success
        })
        .catch(err => {
            setError(`Failed to start training job: ${err.message}`);
        })
        .finally(() => {
            setIsLoading(false);
        });
    };

    return (
        <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
            <h2>Start New Model Training</h2>
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '500px' }}>
                <div>
                    <label htmlFor="asset-select" style={{ display: 'block', marginBottom: '0.5rem' }}>Asset</label>
                    <select 
                        id="asset-select" 
                        value={selectedAsset} 
                        onChange={e => setSelectedAsset(e.target.value)} 
                        required
                        style={{ width: '100%', padding: '0.5rem' }}
                    >
                        {assets.map(asset => (
                            <option key={asset.id} value={asset.id}>{asset.name} ({asset.id})</option>
                        ))}
                    </select>
                </div>
                <div>
                    <label htmlFor="algorithm-select" style={{ display: 'block', marginBottom: '0.5rem' }}>Algorithm</label>
                    <select 
                        id="algorithm-select" 
                        value={selectedAlgorithm} 
                        onChange={e => setSelectedAlgorithm(e.target.value)} 
                        required
                        style={{ width: '100%', padding: '0.5rem' }}
                    >
                        {algorithms.map(algo => (
                            <option key={algo} value={algo}>{algo}</option>
                        ))}
                    </select>
                </div>
                <div>
                    <label htmlFor="s3-path" style={{ display: 'block', marginBottom: '0.5rem' }}>S3 Data Path (Key)</label>
                    <input 
                        id="s3-path"
                        type="text" 
                        value={s3DataPath} 
                        onChange={e => setS3DataPath(e.target.value)} 
                        placeholder="e.g., training-data/production_line_a_2025.csv"
                        required
                        style={{ width: '100%', padding: '0.5rem' }}
                    />
                </div>
                <div>
                    <label htmlFor="n-epochs" style={{ display: 'block', marginBottom: '0.5rem' }}>Number of Epochs</label>
                    <input 
                        id="n-epochs"
                        type="number" 
                        value={nEpochs} 
                        onChange={e => setNEpochs(parseInt(e.target.value, 10))}
                        min="1" // Minimum value
                        max="200" // Maximum value to prevent excessively long training
                        required
                        style={{ width: '100%', padding: '0.5rem' }}
                    />
                    <small style={{ color: '#666', fontSize: '0.8rem', marginTop: '0.25rem' }}>
                        Recommended for neural networks (TiDE, LSTM, TFT): 20-100. LightGBM does not use epochs.
                    </small>
                </div>
                <button type="submit" disabled={isLoading || assets.length === 0} style={{ padding: '0.75rem', cursor: 'pointer' }}>
                    {isLoading ? 'Starting Job...' : 'Start Training Job'}
                </button>
            </form>
            {message && <p style={{ color: 'green', marginTop: '1rem' }}>{message}</p>}
            {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}
        </div>
    );
}

export default ModelTraining;