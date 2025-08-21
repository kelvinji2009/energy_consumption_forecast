import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import TaskStatus from './components/TaskStatus';
import ApiKeyPromptModal from './components/ApiKeyPromptModal';
import './App.css';

// 现代化欢迎页面组件
function WelcomePage() {
  return (
    <div className="modern-card">
      <div className="welcome-section">
        <h2>🚀 能耗预测与异常检测系统</h2>
        <p>
          基于先进机器学习算法的工业能耗管理平台，提供实时预测、异常检测和智能分析服务。
          支持多种预测模型，助力企业实现能源优化和成本控制。
        </p>
        
        <div className="feature-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>智能预测</h3>
            <p>支持 LightGBM、TFT、LSTM、TiDE 等多种先进算法，提供高精度能耗预测</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3>异常检测</h3>
            <p>基于 QuantileDetector 的实时异常监测，及时发现能耗异常并预警</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>异步训练</h3>
            <p>后台异步模型训练，支持大规模数据处理，不影响系统正常运行</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>精准管理</h3>
            <p>多资产管理，支持不同产线和车间的独立建模和预测分析</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">📈</div>
            <h3>可视化分析</h3>
            <p>直观的图表展示，实时监控预测结果和异常检测状态</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3>安全可靠</h3>
            <p>API 密钥认证，容器化部署，确保数据安全和系统稳定性</p>
          </div>
        </div>
      </div>
    </div>
  );
}

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
          <h1>🔋 能耗预测与异常检测管理系统</h1>
          <nav>
            <ul>
              <li><Link to="/">🏠 首页</Link></li>
              <li><Link to="/forecast">📈 能耗预测</Link></li>
              <li><Link to="/anomaly-detection">🚨 异常检测</Link></li>
              <li><Link to="/training">🎯 模型训练</Link></li>
              <li><Link to="/models">🤖 模型管理</Link></li>
              <li><Link to="/assets">🏭 资产管理</Link></li>
              <li><Link to="/api-keys">🔑 API 密钥</Link></li>
            </ul>
          </nav>
        </header>
        <main className="container">
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/forecast" element={
              <div className="modern-card">
                <ForecastView />
              </div>
            } />
            <Route path="/anomaly-detection" element={
              <div className="modern-card">
                <AnomalyDetectionView />
              </div>
            } />
            <Route path="/training" element={
              <div className="modern-card">
                <ModelTraining setActiveTask={setActiveTask} activeTask={activeTask} />
              </div>
            } />
            <Route path="/assets" element={
              <div className="modern-card">
                <AssetList />
              </div>
            } />
            <Route path="/models" element={
              <div className="modern-card">
                <ModelList />
              </div>
            } />
            <Route path="/api-keys" element={
              <div className="modern-card">
                <ApiKeyList />
              </div>
            } />
          </Routes>
          {activeTask && (
            <div className="task-status-float">
              <TaskStatus task={activeTask} onTaskComplete={handleTaskComplete} />
            </div>
          )}
        </main>
      </div>
    </Router>
  );
}

export default App;