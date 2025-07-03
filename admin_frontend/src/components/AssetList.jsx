import React, { useState, useEffect } from 'react';

function AssetList() {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        // 注意：这里直接调用本地后端API
        const response = await fetch('http://127.0.0.1:8000/admin/assets');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAssets(data);
      } catch (error) {
        console.error("Error fetching assets:", error);
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    fetchAssets();
  }, []); // 空依赖数组表示只在组件挂载时运行一次

  if (loading) {
    return <div className="loading-message">加载资产中...</div>;
  }

  if (error) {
    return <div className="error-message">加载资产失败: {error.message}</div>;
  }

  return (
    <div>
      <h2>资产列表</h2>
      {assets.length === 0 ? (
        <p>没有找到任何资产。</p>
      ) : (
        <ul>
          {assets.map(asset => (
            <li key={asset.id}>
              <span>ID: {asset.id}</span>
              <span>名称: {asset.name}</span>
              <span>描述: {asset.description || '无'}</span>
              {/* 未来可以添加编辑/删除按钮 */}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default AssetList;
