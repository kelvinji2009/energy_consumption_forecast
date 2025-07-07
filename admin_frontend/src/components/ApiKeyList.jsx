import React, { useState, useEffect, useCallback } from 'react';
import apiClient from '../apiClient'; // Import the centralized API client

function ApiKeyList() { // Remove apiKey prop
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newKeyDescription, setNewKeyDescription] = useState('');
  const [generatedKey, setGeneratedKey] = useState(null); // To display the newly generated key

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
    setGeneratedKey(null); // Clear previous key
    try {
      const data = await apiClient('/admin/api-keys', {
        method: 'POST',
        body: JSON.stringify({ description: newKeyDescription }),
      });
      setGeneratedKey(data.key); // Display the new key
      setNewKeyDescription(''); // Clear input
      fetchApiKeys(); // Refresh the list
    } catch (error) {
      console.error("Error creating API key:", error);
      setError(error);
    }
  };

  const handleDeleteKey = async (keyId) => {
    if (window.confirm('Are you sure you want to delete this API key?')) {
      try {
        await apiClient(`/admin/api-keys/${keyId}`, {
          method: 'DELETE',
        });
        fetchApiKeys(); // Refresh the list
      } catch (error) {
        console.error("Error deleting API key:", error);
        setError(error);
      }
    }
  };

  // NOTE: The /toggle-active endpoint was not implemented in the backend API.
  // This function is commented out to prevent errors. If the endpoint is added later,
  // it can be re-enabled.
  /*
  const handleToggleActive = async (keyId) => {
    try {
      await apiClient(`/admin/api-keys/${keyId}/toggle-active`, {
        method: 'PUT',
      });
      fetchApiKeys(); // Refresh the list
    } catch (error) {
      console.error("Error toggling API key status:", error);
      setError(error);
    }
  };
  */

  if (loading) {
    return <div className="loading-message">Loading API Keys...</div>;
  }

  if (error) {
    return <div className="error-message">Failed to load API Keys: {error.message}</div>;
  }

  return (
    <div>
      <h2>API Key Management</h2>

      <h3>Create New Key</h3>
      <form onSubmit={handleCreateKey} style={{ marginBottom: '20px' }}>
        <input
          type="text"
          placeholder="Key Description (optional)"
          value={newKeyDescription}
          onChange={(e) => setNewKeyDescription(e.target.value)}
          style={{ padding: '8px', marginRight: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '8px 15px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Create Key</button>
      </form>
      {generatedKey && (
        <div style={{ backgroundColor: '#e9f7ef', border: '1px solid #d0e9da', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
          <strong>New key generated (please save, it is shown only once):</strong><br/>
          <code>{generatedKey}</code>
        </div>
      )}

      <h3>Existing Keys</h3>
      {apiKeys.length === 0 ? (
        <p>No API keys found.</p>
      ) : (
        <ul>
          {apiKeys.map(key => (
            <li key={key.id} style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>ID: {key.id}</span>
                <span>Status: {key.is_active ? 'Active' : 'Inactive'}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>Description: {key.description || 'None'}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>Created: {new Date(key.created_at).toLocaleString()}</span>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                {/* <button onClick={() => handleToggleActive(key.id)}>Toggle Active</button> */}
                <button 
                  onClick={() => handleDeleteKey(key.id)}
                  style={{ padding: '8px 15px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ApiKeyList;
