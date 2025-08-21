import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function AssetList() {
  const { t } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [currentAsset, setCurrentAsset] = useState({ id: '', name: '', description: '' });

  const fetchAssets = async () => {
    setLoading(true);
    try {
      const data = await apiClient('/admin/assets');
      setAssets(data);
      setError(null);
    } catch (err) {
      setError(`${t.assets.fetchError}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAssets();
  }, []);

  const handleOpen = (asset = null) => {
    if (asset) {
      setIsEditing(true);
      setCurrentAsset(asset);
    } else {
      setIsEditing(false);
      setCurrentAsset({ id: '', name: '', description: '' });
    }
    setShowModal(true);
    setError(null);
  };

  const handleClose = () => {
    setShowModal(false);
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setCurrentAsset(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async () => {
    try {
      if (isEditing) {
        await apiClient(`/admin/assets/${currentAsset.id}`, {
          method: 'PUT',
          body: JSON.stringify({ name: currentAsset.name, description: currentAsset.description }),
        });
      } else {
        await apiClient('/admin/assets', {
          method: 'POST',
          body: JSON.stringify(currentAsset),
        });
      }
      fetchAssets();
      handleClose();
    } catch (err) {
      setError(`${t.assets.saveError}: ${err.message}`);
    }
  };

  const handleDelete = async (assetId) => {
    if (window.confirm(t.assets.deleteConfirm.replace('{id}', assetId))) {
      try {
        await apiClient(`/admin/assets/${assetId}`, { method: 'DELETE' });
        fetchAssets();
      } catch (err) {
        alert(`${t.assets.deleteError}: ${err.message}`);
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
        ⏳ {t.assets.loading}
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
        {error}
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
            🏭 {t.assets.title}
          </h2>
          <button
            onClick={() => handleOpen()}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '25px',
              fontSize: '0.9rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            ➕ {t.assets.createNew}
          </button>
        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.8)',
          borderRadius: '16px',
          overflow: 'hidden',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'rgba(102, 126, 234, 0.1)' }}>
                <th style={{
                  padding: '1rem',
                  textAlign: 'left',
                  fontWeight: '600',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {t.assets.id}
                </th>
                <th style={{
                  padding: '1rem',
                  textAlign: 'left',
                  fontWeight: '600',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {t.assets.name}
                </th>
                <th style={{
                  padding: '1rem',
                  textAlign: 'left',
                  fontWeight: '600',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {t.assets.description}
                </th>
                <th style={{
                  padding: '1rem',
                  textAlign: 'left',
                  fontWeight: '600',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {t.assets.modelCount}
                </th>
                <th style={{
                  padding: '1rem',
                  textAlign: 'right',
                  fontWeight: '600',
                  color: '#4a5568',
                  fontSize: '0.9rem'
                }}>
                  {t.assets.actions}
                </th>
              </tr>
            </thead>
            <tbody>
              {assets.map((asset, index) => (
                <tr
                  key={asset.id}
                  style={{
                    borderBottom: index < assets.length - 1 ? '1px solid #e2e8f0' : 'none',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.target.parentElement.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.parentElement.style.backgroundColor = 'transparent';
                  }}
                >
                  <td style={{ padding: '1rem', color: '#2d3748', fontWeight: '500' }}>
                    {asset.id}
                  </td>
                  <td style={{ padding: '1rem', color: '#2d3748' }}>
                    {asset.name}
                  </td>
                  <td style={{ padding: '1rem', color: '#718096' }}>
                    {asset.description || 'N/A'}
                  </td>
                  <td style={{ padding: '1rem', color: '#2d3748' }}>
                    <span style={{
                      background: asset.model_count > 0 ? 'rgba(56, 161, 105, 0.1)' : 'rgba(160, 174, 192, 0.1)',
                      color: asset.model_count > 0 ? '#38a169' : '#a0aec0',
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      fontWeight: '600'
                    }}>
                      {asset.model_count}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', textAlign: 'right' }}>
                    <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => handleOpen(asset)}
                        style={{
                          padding: '0.5rem',
                          background: 'rgba(102, 126, 234, 0.1)',
                          color: '#667eea',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          fontSize: '1rem'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = '#667eea';
                          e.target.style.color = 'white';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(102, 126, 234, 0.1)';
                          e.target.style.color = '#667eea';
                        }}
                      >
                        ✏️
                      </button>
                      <button
                        onClick={() => handleDelete(asset.id)}
                        disabled={asset.model_count > 0}
                        style={{
                          padding: '0.5rem',
                          background: asset.model_count > 0 ? 'rgba(160, 174, 192, 0.1)' : 'rgba(229, 62, 62, 0.1)',
                          color: asset.model_count > 0 ? '#a0aec0' : '#e53e3e',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: asset.model_count > 0 ? 'not-allowed' : 'pointer',
                          transition: 'all 0.2s ease',
                          fontSize: '1rem'
                        }}
                        onMouseEnter={(e) => {
                          if (asset.model_count === 0) {
                            e.target.style.background = '#e53e3e';
                            e.target.style.color = 'white';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (asset.model_count === 0) {
                            e.target.style.background = 'rgba(229, 62, 62, 0.1)';
                            e.target.style.color = '#e53e3e';
                          }
                        }}
                      >
                        🗑️
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Modal */}
        {showModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              width: '90%',
              maxWidth: '500px',
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)'
            }}>
              <h3 style={{
                margin: '0 0 1.5rem 0',
                color: '#2d3748',
                fontSize: '1.3rem',
                fontWeight: '600'
              }}>
                {isEditing ? t.assets.editAsset : t.assets.createAsset}
              </h3>

              {error && (
                <div style={{
                  padding: '1rem',
                  background: 'rgba(254, 178, 178, 0.9)',
                  color: '#c53030',
                  borderRadius: '8px',
                  marginBottom: '1rem',
                  fontSize: '0.9rem'
                }}>
                  {error}
                </div>
              )}

              <div style={{ display: 'grid', gap: '1rem' }}>
                <div>
                  <label style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontWeight: '500',
                    color: '#4a5568',
                    fontSize: '0.9rem'
                  }}>
                    {t.assets.assetId}:
                  </label>
                  <input
                    type="text"
                    name="id"
                    value={currentAsset.id}
                    onChange={handleChange}
                    disabled={isEditing}
                    placeholder="production_line_C"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '2px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      background: isEditing ? '#f7fafc' : 'white',
                      color: isEditing ? '#a0aec0' : '#2d3748'
                    }}
                  />
                  {isEditing && (
                    <p style={{ fontSize: '0.8rem', color: '#718096', margin: '0.25rem 0 0 0' }}>
                      {t.assets.idCannotChange}
                    </p>
                  )}
                </div>

                <div>
                  <label style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontWeight: '500',
                    color: '#4a5568',
                    fontSize: '0.9rem'
                  }}>
                    {t.assets.assetName}:
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={currentAsset.name}
                    onChange={handleChange}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '2px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      background: 'white'
                    }}
                  />
                </div>

                <div>
                  <label style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontWeight: '500',
                    color: '#4a5568',
                    fontSize: '0.9rem'
                  }}>
                    {t.assets.assetDescription}:
                  </label>
                  <textarea
                    name="description"
                    value={currentAsset.description}
                    onChange={handleChange}
                    rows="4"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '2px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      background: 'white',
                      resize: 'vertical'
                    }}
                  />
                </div>
              </div>

              <div style={{
                display: 'flex',
                gap: '1rem',
                justifyContent: 'flex-end',
                marginTop: '2rem'
              }}>
                <button
                  onClick={handleClose}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: 'rgba(160, 174, 192, 0.2)',
                    color: '#4a5568',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '0.9rem',
                    fontWeight: '500',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {t.assets.cancel}
                </button>
                <button
                  onClick={handleSubmit}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '0.9rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)'
                  }}
                >
                  {isEditing ? t.assets.saveChanges : t.assets.create}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AssetList;