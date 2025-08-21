import React, { useState, useEffect, useCallback } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function ApiKeyList() {
  const { t } = useLanguage();
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newKeyDescription, setNewKeyDescription] = useState('');
  const [generatedKey, setGeneratedKey] = useState(null);

  const fetchApiKeys = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient('/admin/api-keys');
      setApiKeys(data);
    } catch (error) {
      console.error("Error fetching API keys:", error);
      setError(error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchApiKeys();
  }, [fetchApiKeys]);

  const handleCreateKey = async (e) => {
    e.preventDefault();
    setGeneratedKey(null);
    try {
      const data = await apiClient('/admin/api-keys', {
        method: 'POST',
        body: JSON.stringify({ description: newKeyDescription }),
      });
      setGeneratedKey(data.key);
      setNewKeyDescription('');
      fetchApiKeys();
    } catch (error) {
      console.error("Error creating API key:", error);
      setError(error);
    }
  };

  const handleDeleteKey = async (keyId) => {
    if (window.confirm(t.apiKeys.deleteConfirm)) {
      try {
        await apiClient(`/admin/api-keys/${keyId}`, {
          method: 'DELETE',
        });
        fetchApiKeys();
      } catch (error) {
        console.error("Error deleting API key:", error);
        setError(error);
      }
    }
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
        ⏳ {t.apiKeys.loading}
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
        {t.apiKeys.loadError}: {error.message}
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
        <h2 style={{
          color: '#4a5568',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          🔑 {t.apiKeys.title}
        </h2>

        {/* Create New Key Section */}
        <div style={{
          background: 'rgba(102, 126, 234, 0.05)',
          borderRadius: '16px',
          padding: '1.5rem',
          marginBottom: '2rem',
          border: '1px solid rgba(102, 126, 234, 0.1)'
        }}>
          <h3 style={{
            color: '#4a5568',
            marginBottom: '1rem',
            fontSize: '1.1rem',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            ➕ {t.apiKeys.createNew}
          </h3>
          
          <form onSubmit={handleCreateKey} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
            <div style={{ flex: 1 }}>
              <label style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontWeight: '500',
                color: '#4a5568',
                fontSize: '0.9rem'
              }}>
                {t.apiKeys.description}:
              </label>
              <input
                type="text"
                placeholder={t.apiKeys.descriptionPlaceholder}
                value={newKeyDescription}
                onChange={(e) => setNewKeyDescription(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid #e2e8f0',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  background: 'white',
                  transition: 'border-color 0.2s ease'
                }}
                onFocus={(e) => e.target.style.borderColor = '#667eea'}
                onBlur={(e) => e.target.style.borderColor = '#e2e8f0'}
              />
            </div>
            <button
              type="submit"
              style={{
                padding: '0.75rem 1.5rem',
                background: 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '0.9rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 15px rgba(72, 187, 120, 0.4)',
                whiteSpace: 'nowrap'
              }}
            >
              {t.apiKeys.createKey}
            </button>
          </form>
        </div>

        {/* Generated Key Display */}
        {generatedKey && (
          <div style={{
            background: 'rgba(56, 161, 105, 0.1)',
            border: '1px solid rgba(56, 161, 105, 0.3)',
            padding: '1.5rem',
            borderRadius: '12px',
            marginBottom: '2rem'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1rem',
              color: '#38a169',
              fontWeight: '600'
            }}>
              ✅ {t.apiKeys.keyGenerated}
            </div>
            <div style={{
              background: 'white',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px solid rgba(56, 161, 105, 0.2)',
              fontFamily: 'monospace',
              fontSize: '0.9rem',
              wordBreak: 'break-all',
              color: '#2d3748'
            }}>
              {generatedKey}
            </div>
            <p style={{
              margin: '0.5rem 0 0 0',
              fontSize: '0.8rem',
              color: '#718096'
            }}>
              {t.apiKeys.saveWarning}
            </p>
          </div>
        )}

        {/* Existing Keys Section */}
        <div>
          <h3 style={{
            color: '#4a5568',
            marginBottom: '1rem',
            fontSize: '1.1rem',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📋 {t.apiKeys.existingKeys}
          </h3>

          {apiKeys.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '3rem',
              color: '#718096',
              fontSize: '1rem'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔑</div>
              <p>{t.apiKeys.noKeys}</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: '1rem' }}>
              {apiKeys.map(key => (
                <div
                  key={key.id}
                  style={{
                    background: 'rgba(255, 255, 255, 0.8)',
                    border: '1px solid rgba(226, 232, 240, 0.8)',
                    borderRadius: '12px',
                    padding: '1.5rem',
                    transition: 'all 0.3s ease'
                  }}
                >
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: '1rem'
                  }}>
                    <div>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '1rem',
                        marginBottom: '0.5rem'
                      }}>
                        <span style={{
                          fontWeight: '600',
                          color: '#2d3748',
                          fontSize: '1rem'
                        }}>
                          ID: {key.id}
                        </span>
                        <span style={{
                          background: key.is_active ? 'rgba(56, 161, 105, 0.1)' : 'rgba(160, 174, 192, 0.1)',
                          color: key.is_active ? '#38a169' : '#a0aec0',
                          padding: '0.25rem 0.75rem',
                          borderRadius: '20px',
                          fontSize: '0.8rem',
                          fontWeight: '600'
                        }}>
                          {key.is_active ? t.apiKeys.status.active : t.apiKeys.status.inactive}
                        </span>
                      </div>
                      <div style={{ color: '#718096', fontSize: '0.9rem', marginBottom: '0.25rem' }}>
                        <strong>{t.apiKeys.description}:</strong> {key.description || t.apiKeys.noDescription}
                      </div>
                      <div style={{ color: '#718096', fontSize: '0.9rem' }}>
                        <strong>{t.apiKeys.created}:</strong> {new Date(key.created_at).toLocaleString()}
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteKey(key.id)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: 'rgba(229, 62, 62, 0.1)',
                        color: '#e53e3e',
                        border: 'none',
                        borderRadius: '8px',
                        fontSize: '0.8rem',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = '#e53e3e';
                        e.target.style.color = 'white';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = 'rgba(229, 62, 62, 0.1)';
                        e.target.style.color = '#e53e3e';
                      }}
                    >
                      🗑️ {t.apiKeys.delete}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ApiKeyList;