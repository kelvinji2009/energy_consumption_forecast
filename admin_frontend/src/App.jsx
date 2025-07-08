import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import './App.css';

const API_KEY_STORAGE_KEY = 'app-api-key';

function App() {
  // State to hold the API key, initialized from localStorage or empty string
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
  });

  // Effect to update localStorage whenever the apiKey state changes
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
    } else {
      // If key is cleared, remove it from storage
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  }, [apiKey]);

  return (
    <Router>
      <div className="App">
        <header className="header">
          <h1>能耗预测与异常检测管理系统</h1>
          <nav>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', justifyContent: 'center' }}>
              <li style={{ margin: '0 15px' }}>
                <Link to="/" style={{ color: '#61dafb', textDecoration: 'none' }}>首页</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/forecast" style={{ color: '#61dafb', textDecoration: 'none' }}>能耗预测</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/anomaly-detection" style={{ color: '#61dafb', textDecoration: 'none' }}>异常检测</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/training" style={{ color: '#61dafb', textDecoration: 'none' }}>模型训练</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/models" style={{ color: '#61dafb', textDecoration: 'none' }}>模型管理</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/assets" style={{ color: '#61dafb', textDecoration: 'none' }}>资产管理</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/api-keys" style={{ color: '#61dafb', textDecoration: 'none' }}>API 密钥</Link>
              </li>
            </ul>
          </nav>
          <div style={{ padding: '10px', backgroundColor: '#333', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <label htmlFor="global-api-key" style={{ marginRight: '10px' }}>API Key:</label>
            <input 
              id="global-api-key"
              type="password" 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API Key and press Enter"
              style={{ padding: '5px', width: '300px' }}
            />
          </div>
        </header>
        <main className="container">
          {apiKey ? (
            <Routes>
              <Route path="/" element={<h2>欢迎来到管理面板</h2>} />
              <Route path="/forecast" element={<ForecastView />} />
              <Route path="/anomaly-detection" element={<AnomalyDetectionView />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/assets" element={<AssetList />} />
              <Route path="/models" element={<ModelList />} />
              <Route path="/api-keys" element={<ApiKeyList />} />
            </Routes>
          ) : (
            <div style={{ textAlign: 'center', marginTop: '50px' }}>
              <h2>请输入API密钥以继续</h2>
              <p>请在页面顶部的输入框中提供您的API密钥以访问管理面板。</p>
            </div>
          )}
        </main>
      </div>
    </Router>
  );
}

export default App;
