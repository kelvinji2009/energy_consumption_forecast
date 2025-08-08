import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import TaskStatus from './components/TaskStatus';
import ApiKeyPromptModal from './components/ApiKeyPromptModal'; // Import the new modal
import './App.css';

function App() {
  const [activeTask, setActiveTask] = useState(null);
  const [isApiKeyModalOpen, setApiKeyModalOpen] = useState(false);

  // Effect to listen for global API key requests from the apiClient
  useEffect(() => {
    const handleRequest = () => {
      setApiKeyModalOpen(true);
    };

    window.addEventListener('request-api-key', handleRequest);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener('request-api-key', handleRequest);
    };
  }, []);

  const handleTaskComplete = () => {
    setTimeout(() => {
      setActiveTask(null);
    }, 5000);
  };

  return (
    <Router>
      {isApiKeyModalOpen && (
        <ApiKeyPromptModal onKeySubmit={() => setApiKeyModalOpen(false)} />
      )}
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
        </header>
        <main className="container">
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
        </main>
      </div>
    </Router>
  );
}

export default App;