import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';

function AssetList() {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const data = await apiClient('/admin/assets');
        setAssets(data);
      } catch (error) {
        console.error("Error fetching assets:", error);
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    fetchAssets();
  }, []); // Empty dependency array means this runs once on mount

  if (loading) {
    return <div className="loading-message">Loading Assets...</div>;
  }

  if (error) {
    return <div className="error-message">Failed to load assets: {error.message}</div>;
  }

  return (
    <div>
      <h2>Asset List</h2>
      {assets.length === 0 ? (
        <p>No assets found.</p>
      ) : (
        <ul>
          {assets.map(asset => (
            <li key={asset.id}>
              <span>ID: {asset.id}</span>
              <span>Name: {asset.name}</span>
              <span>Description: {asset.description || 'None'}</span>
              {/* Future functionality: Edit/Delete buttons */}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default AssetList;
