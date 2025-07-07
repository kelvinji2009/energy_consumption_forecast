import React, { useState, useEffect } from 'react';

function ModelList() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/admin/models');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModels(data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))); // Sort by creation date
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchModels, 5000); // Refresh every 5 seconds
    fetchModels(); // Initial fetch
    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  const getStatusStyle = (status) => {
    switch (status) {
      case 'COMPLETED':
        return { color: 'green', fontWeight: 'bold' };
      case 'TRAINING':
        return { color: 'orange', fontWeight: 'bold' };
      case 'PENDING':
        return { color: '#6c757d', fontWeight: 'bold' };
      case 'FAILED':
        return { color: 'red', fontWeight: 'bold' };
      default:
        return {};
    }
  };

  if (loading && models.length === 0) {
    return <div className="loading-message">Loading models...</div>;
  }

  if (error) {
    return <div className="error-message">Failed to load models: {error}</div>;
  }

  return (
    <div>
      <h2>Model List</h2>
      <button onClick={fetchModels} disabled={loading} style={{ marginBottom: '1rem' }}>
        {loading ? 'Refreshing...' : 'Refresh List'}
      </button>
      {models.length === 0 ? (
        <p>No models found. You can start a new training job on the 'Model Training' page.</p>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0 }}>
          {models.map(model => (
            <li key={model.id} style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '1rem', marginBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                <strong>Model ID: {model.id} (Asset: {model.asset_id})</strong>
                <span style={getStatusStyle(model.status)}>{model.status}</span>
              </div>
              <div><strong>Type:</strong> {model.model_type}</div>
              <div><strong>Version:</strong> {model.model_version || 'N/A'}</div>
              <div><strong>Created:</strong> {new Date(model.created_at).toLocaleString()}</div>
              {model.metrics && (
                <div><strong>Metrics (MAPE):</strong> {model.metrics.mape ? `${model.metrics.mape.toFixed(2)}%` : 'N/A'}</div>
              )}
              {model.model_path && (
                <div style={{ wordBreak: 'break-all' }}><strong>S3 Path:</strong> {model.model_path}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ModelList;
