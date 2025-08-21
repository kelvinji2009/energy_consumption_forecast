import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import TaskStatus from './components/TaskStatus';
import ApiKeyPromptModal from './components/ApiKeyPromptModal';
import LanguageToggle from './components/LanguageToggle';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import './App.css';

// 导航组件，处理高亮显示
function Navigation({ safeT }) {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: safeT('nav.home') },
    { path: '/forecast', label: safeT('nav.forecast') },
    { path: '/anomaly-detection', label: safeT('nav.anomaly') },
    { path: '/training', label: safeT('nav.training') },
    { path: '/models', label: safeT('nav.models') },
    { path: '/assets', label: safeT('nav.assets') },
    { path: '/api-keys', label: safeT('nav.apiKeys') }
  ];

  return (
    <nav>
      <ul>
        {navItems.map(item => (
          <li key={item.path}>
            <Link 
              to={item.path}
              style={{
                color: location.pathname === item.path ? '#667eea' : 'inherit',
                fontWeight: location.pathname === item.path ? '600' : 'normal',
                textDecoration: location.pathname === item.path ? 'underline' : 'none',
                textUnderlineOffset: '4px'
              }}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function AppContent() {
  const { t } = useLanguage();
  const [activeTask, setActiveTask] = useState(null);
  const [isApiKeyModalOpen, setApiKeyModalOpen] = useState(false);

  // 安全检查：如果t函数不存在，提供默认值
  const safeT = (key) => {
    if (typeof t === 'function') {
      return t(key);
    }
    // 提供默认的中文文本
    const defaultTexts = {
      'nav.home': '首页',
      'nav.forecast': '能耗预测',
      'nav.anomaly': '异常检测',
      'nav.training': '模型训练',
      'nav.models': '模型管理',
      'nav.assets': '资产管理',
      'nav.apiKeys': 'API 密钥',
      'home.title': '能耗预测与异常检测管理系统',
      'home.subtitle': '基于机器学习的工业能耗智能分析平台',
      'home.features.prediction.title': '智能预测',
      'home.features.prediction.desc': '支持多种算法的能耗预测模型',
      'home.features.anomaly.title': '异常检测',
      'home.features.anomaly.desc': '实时识别能耗异常模式',
      'home.features.async.title': '异步处理',
      'home.features.async.desc': '高效的后台任务处理',
      'home.features.management.title': '资产管理',
      'home.features.management.desc': '统一的设备和资产管理',
      'home.features.visualization.title': '数据可视化',
      'home.features.visualization.desc': '直观的图表和报告',
      'home.features.security.title': '安全认证',
      'home.features.security.desc': 'API密钥和权限管理'
    };
    return defaultTexts[key] || key;
  };

  // 现代化欢迎页面组件 - 在AppContent内部定义
  const WelcomePage = () => {
    return (
      <div className="modern-card">
        <div className="welcome-section">
          <h2>{safeT('home.title')}</h2>
          <p>{safeT('home.subtitle')}</p>
          
          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>{safeT('home.features.prediction.title')}</h3>
              <p>{safeT('home.features.prediction.desc')}</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h3>{safeT('home.features.anomaly.title')}</h3>
              <p>{safeT('home.features.anomaly.desc')}</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>{safeT('home.features.async.title')}</h3>
              <p>{safeT('home.features.async.desc')}</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>{safeT('home.features.management.title')}</h3>
              <p>{safeT('home.features.management.desc')}</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">📈</div>
              <h3>{safeT('home.features.visualization.title')}</h3>
              <p>{safeT('home.features.visualization.desc')}</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>{safeT('home.features.security.title')}</h3>
              <p>{safeT('home.features.security.desc')}</p>
            </div>
          </div>
        </div>
      </div>
    );
  };

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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h1>🔋 能耗预测系统</h1>
            <LanguageToggle />
          </div>
          <Navigation safeT={safeT} />
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

function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;