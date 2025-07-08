import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import TaskStatus from './components/TaskStatus'; // Import the TaskStatus component
import './App.css';

const API_KEY_STORAGE_KEY = 'app-api-key';

function App() {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(API_KEY_STORAGE_KEY) || '');
  const [activeTask, setActiveTask] = useState(null); // State for the active task

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
    } else {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  }, [apiKey]);

  const handleTaskComplete = () => {
    // After a delay, clear the task so the user can start a new one
    setTimeout(() => {
      setActiveTask(null);
    }, 5000); // Keep the status card for 5 seconds after completion
  };

  return (
    <Router>
      <div className="App">
        <header className="header">
          <h1>能耗预测与异常检测管理系统</h1>
          <nav>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', justifyContent: 'center' }}>
              <li style={{ margin: '0 15px' }}><Link to="/" style={{ color: '#61dafb', textDecoration: 'none' }}>首页</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/forecast" style={{ color: '#61dafb', textDecoration: 'none' }}>能耗预测</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/anomaly-detection" style={{ color: '#61dafb', textDecoration: 'none' }}>异常检测</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/training" style={{ color: '#61dafb', textDecoration: 'none' }}>模型训练</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/models" style={{ color: '#61dafb', textDecoration: 'none' }}>模型管理</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/assets" style={{ color: '#61dafb', textDecoration: 'none' }}>资产管理</Link></li>
              <li style={{ margin: '0 15px' }}><Link to="/api-keys" style={{ color: '#61dafb', textDecoration: 'none' }}>API 密钥</Link></li>
            </ul>
          </nav>
          <div style={{ padding: '10px', backgroundColor: '#333', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <label htmlFor="global-api-key" style={{ marginRight: '10px' }}>API Key:</label>
            <input 
              id="global-api-key"
              type="password" 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API Key to access the system"
              style={{ padding: '5px', width: '300px' }}
            />
          </div>
        </header>
        <main className="container">
          {apiKey ? (
            <>
              <Routes>
                <Route path="/" element={<h2>欢迎来到管理面板</h2>} />
                <Route path="/forecast" element={<ForecastView />} />
                <Route path="/anomaly-detection" element={<AnomalyDetectionView />} />
                <Route path="/training" element={<ModelTraining setActiveTask={setActiveTask} activeTask={activeTask} />} />
                <Route path="/assets" element={<AssetList />} />
                <Route path="/models" element={<ModelList />} />
                <Route path="/api-keys" element={<ApiKeyList />} />
              </Routes>
              {activeTask && (
                <div style={{ position: 'fixed', bottom: '1rem', right: '1rem', zIndex: 1000, minWidth: '300px' }}>
                  <TaskStatus task={activeTask} onTaskComplete={handleTaskComplete} />
                </div>
              )}
            </>
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
