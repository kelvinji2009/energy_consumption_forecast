import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList'; // 导入ApiKeyList
import './App.css';

function App() {
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
                <Link to="/assets" style={{ color: '#61dafb', textDecoration: 'none' }}>资产管理</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/models" style={{ color: '#61dafb', textDecoration: 'none' }}>模型管理</Link>
              </li>
              <li style={{ margin: '0 15px' }}>
                <Link to="/api-keys" style={{ color: '#61dafb', textDecoration: 'none' }}>API 密钥</Link> {/* 新增API密钥链接 */}
              </li>
            </ul>
          </nav>
        </header>
        <main className="container">
          <Routes>
            <Route path="/" element={<h2>欢迎来到管理面板</h2>} />
            <Route path="/assets" element={<AssetList />} />
            <Route path="/models" element={<ModelList />} />
            <Route path="/api-keys" element={<ApiKeyList />} /> {/* 新增API密钥路由 */}
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;