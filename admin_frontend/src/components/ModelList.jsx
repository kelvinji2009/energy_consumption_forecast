import React, { useState, useEffect } from 'react';

function ModelList() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/admin/models');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleActivateModel = async (modelId, assetId) => {
    try {
      const activateResponse = await fetch(`http://127.0.0.1:8000/admin/models/${modelId}/activate`, {
        method: 'PUT',
      });
      if (!activateResponse.ok) {
        throw new Error(`HTTP error! status: ${activateResponse.status}`);
      }
      alert(`模型 ${modelId} 已激活！`);
      fetchModels(); // 刷新列表
    } catch (error) {
      console.error("Error activating model:", error);
      setError(error);
      alert(`激活模型失败: ${error.message}`);
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (window.confirm('确定要删除此模型吗？')) {
      try {
        const response = await fetch(`http://127.0.0.1:8000/admin/models/${modelId}`, {
          method: 'DELETE',
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        alert(`模型 ${modelId} 已删除！`);
        fetchModels(); // 刷新列表
      } catch (error) {
        console.error("Error deleting model:", error);
        setError(error);
        alert(`删除模型失败: ${error.message}`);
      }
    }
  };

  const handleTriggerTraining = async (assetId) => {
    // 使用您提供的真实数据URL
    const realDataUrl = "https://gist.githubusercontent.com/kelvinji2009/4410fb43d5c60b14d808d9f49994507d/raw/0492a9aaa409ac0db87cce6bf10ca0dfbfe296d7/simulated_plant_data.csv";
    if (window.confirm(`确定要为资产 ${assetId} 触发模型训练吗？`)) {
      try {
        const response = await fetch('http://127.0.0.1:8000/admin/train-model', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ asset_id: assetId, data_url: realDataUrl }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        alert(`训练请求已发送: ${result.message} 任务ID: ${result.task_id}`);
      } catch (error) {
        console.error("Error triggering training:", error);
        setError(error);
        alert(`触发训练失败: ${error.message}`);
      }
    }
  };

  if (loading) {
    return <div className="loading-message">加载模型中...</div>;
  }

  if (error) {
    return <div className="error-message">加载模型失败: {error.message}</div>;
  }

  return (
    <div>
      <h2>模型列表</h2>
      {models.length === 0 ? (
        <p>没有找到任何模型。</p>
      ) : (
        <ul>
          {models.map(model => (
            <li key={model.id} style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>资产ID: {model.asset_id}</span>
                <span>版本: {model.version}</span>
                <span>类型: {model.model_type}</span>
                <span>活跃: {model.is_active ? '是' : '否'}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>路径: {model.path}</span>
              </div>
              <div style={{ width: '100%', marginBottom: '5px' }}>
                <span>训练时间: {new Date(model.trained_at).toLocaleString()}</span>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                {!model.is_active && (
                  <button 
                    onClick={() => handleActivateModel(model.id, model.asset_id)}
                    style={{ padding: '8px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                  >
                    激活
                  </button>
                )}
                <button 
                  onClick={() => handleDeleteModel(model.id)}
                  style={{ padding: '8px 15px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                >
                  删除
                </button>
                <button 
                  onClick={() => handleTriggerTraining(model.asset_id)}
                  style={{ padding: '8px 15px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                >
                  触发训练
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ModelList;
