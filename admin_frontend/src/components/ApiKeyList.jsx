import React, { useState, useEffect } from 'react';

function ApiKeyList() {
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newKeyDescription, setNewKeyDescription] = useState('');
  const [generatedKey, setGeneratedKey] = useState(null); // 用于显示新生成的密钥

  const fetchApiKeys = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/admin/api-keys');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setApiKeys(data);
    } catch (error) {
      console.error("Error fetching API keys:", error);
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApiKeys();
  }, []);

  const handleCreateKey = async (e) => {
    e.preventDefault();
    setGeneratedKey(null); // 清除上次生成的密钥
    try {
      const response = await fetch('http://127.0.0.1:8000/admin/api-keys', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description: newKeyDescription }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setGeneratedKey(data.key); // 显示新生成的密钥
      setNewKeyDescription(''); // 清空输入框
      fetchApiKeys(); // 刷新列表
    } catch (error) {
      console.error("Error creating API key:", error);
      setError(error);
    }
  };

  const handleToggleActive = async (keyId, currentStatus) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/admin/api-keys/${keyId}/toggle-active`, {
        method: 'PUT',
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      fetchApiKeys(); // 刷新列表
    } catch (error) {
      console.error("Error toggling API key status:", error);
      setError(error);
    }
  };

  const handleDeleteKey = async (keyId) => {
    if (window.confirm('确定要删除此API密钥吗？')) {
      try {
        const response = await fetch(`http://127.0.0.1:8000/admin/api-keys/${keyId}`, {
          method: 'DELETE',
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        fetchApiKeys(); // 刷新列表
      } catch (error) {
        console.error("Error deleting API key:", error);
        setError(error);
      }
    }
  };

  if (loading) {
    return <div className="loading-message">加载API密钥中...</div>;
  }

  if (error) {
    return <div className="error-message">加载API密钥失败: {error.message}</div>;
  }

  return (
    <div>
      <h2>API 密钥管理</h2>

      <h3>生成新密钥</h3>
      <form onSubmit={handleCreateKey} style={{ marginBottom: '20px' }}>
        <input
          type="text"
          placeholder="密钥描述 (可选)"
          value={newKeyDescription}
          onChange={(e) => setNewKeyDescription(e.target.value)}
          style={{ padding: '8px', marginRight: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '8px 15px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>生成密钥</button>
      </form>
      {generatedKey && (
        <div style={{ backgroundColor: '#e9f7ef', border: '1px solid #d0e9da', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
          <strong>新生成的密钥 (请妥善保存，仅显示一次):</strong><br/>
          <code>{generatedKey}</code>
        </div>
      )}

      <h3>现有密钥</h3>
      {apiKeys.length === 0 ? (
        <p>没有找到任何API密钥。</p>
      ) : (
        <ul>
          {apiKeys.map(key => (
            <li key={key.id} style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>ID: {key.id}</span>
                <span>状态: {key.is_active ? '活跃' : '禁用'}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>描述: {key.description || '无'}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>创建时间: {new Date(key.created_at).toLocaleString()}</span>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button 
                  onClick={() => handleToggleActive(key.id, key.is_active)}
                  style={{
                    padding: '8px 15px',
                    backgroundColor: key.is_active ? '#ffc107' : '#17a2b8',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  {key.is_active ? '禁用' : '启用'}
                </button>
                <button 
                  onClick={() => handleDeleteKey(key.id)}
                  style={{ padding: '8px 15px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                >
                  删除
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
