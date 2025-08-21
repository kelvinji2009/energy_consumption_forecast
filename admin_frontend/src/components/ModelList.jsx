import React, { useState, useEffect, useCallback } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function ModelList() {
  const { t } = useLanguage();
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiClient('/admin/models');
      setModels(data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)));
      setError(null);
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  const getStatusStyle = (status) => {
    switch (status) {
      case 'COMPLETED':
        return { 
          color: '#38a169', 
          background: 'rgba(56, 161, 105, 0.1)',
          padding: '0.25rem 0.75rem',
          borderRadius: '20px',
          fontSize: '0.8rem',
          fontWeight: '600'
        };
      case 'TRAINING':
        return { 
          color: '#ed8936', 
          background: 'rgba(237, 137, 54, 0.1)',
          padding: '0.25rem 0.75rem',
          borderRadius: '20px',
          fontSize: '0.8rem',
          fontWeight: '600'
        };
      case 'PENDING':
        return { 
          color: '#6c757d', 
          background: 'rgba(108, 117, 125, 0.1)',
          padding: '0.25rem 0.75rem',
          borderRadius: '20px',
          fontSize: '0.8rem',
          fontWeight: '600'
        };
      case 'FAILED':
        return { 
          color: '#e53e3e', 
          background: 'rgba(229, 62, 62, 0.1)',
          padding: '0.25rem 0.75rem',
          borderRadius: '20px',
          fontSize: '0.8rem',
          fontWeight: '600'
        };
      default:
        return {};
    }
  };

  const getStatusText = (status) => {
    const statusMap = {
      'COMPLETED': t.models.status.completed,
      'TRAINING': t.models.status.training,
      'PENDING': t.models.status.pending,
      'FAILED': t.models.status.failed
    };
    return statusMap[status] || status;
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '200px',
        color: '#4a5568',
        fontSize: '1.1rem'
      }}>
        ⏳ {t.models.loading}
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem',
        background: 'rgba(254, 178, 178, 0.9)',
        color: '#c53030',
        borderRadius: '12px',
        border: '1px solid #feb2b2',
        textAlign: 'center'
      }}>
        {t.models.loadError}: {error}
      </div>
    );
  }

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
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '2rem'
        }}>
          <h2 style={{
            color: '#4a5568',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '1.5rem',
            fontWeight: '600',
            margin: 0
          }}>
            🤖 {t.models.title}
          </h2>
          <button
            onClick={fetchModels}
            disabled={loading}
            style={{
              padding: '0.75rem 1.5rem',
              background: loading ? '#a0aec0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '25px',
              fontSize: '0.9rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            {loading ? '⏳' : '🔄'} {loading ? t.models.refreshing : t.models.refresh}
          </button>
        </div>

        {models.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '3rem',
            color: '#718096',
            fontSize: '1.1rem'
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🤖</div>
            <p>{t.models.noModels}</p>
            <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              {t.models.startTraining}
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '1rem' }}>
            {models.map(model => (
              <div
                key={model.id}
                style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  border: '1px solid rgba(226, 232, 240, 0.8)',
                  borderRadius: '16px',
                  padding: '1.5rem',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.05)';
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '1rem'
                }}>
                  <div>
                    <h3 style={{
                      margin: 0,
                      color: '#2d3748',
                      fontSize: '1.1rem',
                      fontWeight: '600'
                    }}>
                      {t.models.modelId}: {model.id}
                    </h3>
                    <p style={{
                      margin: '0.25rem 0 0 0',
                      color: '#718096',
                      fontSize: '0.9rem'
                    }}>
                      {t.models.asset}: {model.asset_id}
                    </p>
                  </div>
                  <span style={getStatusStyle(model.status)}>
                    {getStatusText(model.status)}
                  </span>
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '1rem',
                  marginBottom: '1rem'
                }}>
                  <div>
                    <strong style={{ color: '#4a5568', fontSize: '0.9rem' }}>
                      {t.models.type}:
                    </strong>
                    <span style={{ marginLeft: '0.5rem', color: '#2d3748' }}>
                      {model.model_type}
                    </span>
                  </div>
                  <div>
                    <strong style={{ color: '#4a5568', fontSize: '0.9rem' }}>
                      {t.models.version}:
                    </strong>
                    <span style={{ marginLeft: '0.5rem', color: '#2d3748' }}>
                      v{model.model_version || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <strong style={{ color: '#4a5568', fontSize: '0.9rem' }}>
                      {t.models.created}:
                    </strong>
                    <span style={{ marginLeft: '0.5rem', color: '#2d3748' }}>
                      {new Date(model.created_at).toLocaleString()}
                    </span>
                  </div>
                  {model.metrics && (
                    <div>
                      <strong style={{ color: '#4a5568', fontSize: '0.9rem' }}>
                        {t.models.mape}:
                      </strong>
                      <span style={{ 
                        marginLeft: '0.5rem', 
                        color: model.metrics.mape < 0.1 ? '#38a169' : model.metrics.mape < 0.2 ? '#ed8936' : '#e53e3e',
                        fontWeight: '600'
                      }}>
                        {model.metrics.mape ? `${(model.metrics.mape * 100).toFixed(2)}%` : 'N/A'}
                      </span>
                    </div>
                  )}
                </div>

                {model.model_path && (
                  <div style={{
                    background: 'rgba(247, 250, 252, 0.8)',
                    padding: '0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0'
                  }}>
                    <strong style={{ color: '#4a5568', fontSize: '0.8rem' }}>
                      {t.models.s3Path}:
                    </strong>
                    <div style={{
                      marginTop: '0.25rem',
                      fontSize: '0.8rem',
                      color: '#718096',
                      wordBreak: 'break-all',
                      fontFamily: 'monospace'
                    }}>
                      {model.model_path}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelList;