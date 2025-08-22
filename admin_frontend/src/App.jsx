import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, AppBar, Toolbar, Typography, Container, Box, Tabs, Tab, Card, CardContent, Grid, Chip } from '@mui/material';
import { Home, TrendingUp, Warning, ModelTraining as ModelTrainingIcon, Storage, VpnKey, Description } from '@mui/icons-material';
import AssetList from './components/AssetList';
import ModelList from './components/ModelList';
import ApiKeyList from './components/ApiKeyList';
import ForecastView from './components/ForecastView';
import ModelTraining from './components/ModelTraining';
import AnomalyDetectionView from './components/AnomalyDetectionView';
import ApiDocumentation from './components/ApiDocumentation';
import TaskStatus from './components/TaskStatus';
import ApiKeyPromptModal from './components/ApiKeyPromptModal';
import LanguageToggle from './components/LanguageToggle';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import theme from './theme';

// 导航组件，使用Material UI Tabs
function Navigation() {
  const location = useLocation();
  const { language, t } = useLanguage(); // 直接从上下文获取语言和翻译函数
  
  // 添加语言依赖，确保语言变化时重新渲染
  const navItems = React.useMemo(() => [
    { path: '/', label: t?.nav?.home || '首页', icon: <Home /> },
    { path: '/forecast', label: t?.nav?.forecast || '能耗预测', icon: <TrendingUp /> },
    { path: '/anomaly-detection', label: t?.nav?.anomaly || '异常检测', icon: <Warning /> },
    { path: '/training', label: t?.nav?.training || '模型训练', icon: <ModelTrainingIcon /> },
    { path: '/models', label: t?.nav?.models || '模型管理', icon: <Storage /> },
    { path: '/assets', label: t?.nav?.assets || '资产管理', icon: <Storage /> },
    { path: '/api-keys', label: t?.nav?.apiKeys || 'API 密钥', icon: <VpnKey /> },
    { path: '/api-docs', label: t?.nav?.apiDocs || 'API 文档', icon: <Description /> }
  ], [t, language]); // 添加language和t作为依赖项

  const currentTabIndex = navItems.findIndex(item => item.path === location.pathname);

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', mt: 1 }}>
      <Tabs 
        value={currentTabIndex >= 0 ? currentTabIndex : false}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          '& .MuiTab-root': {
            minHeight: 48,
            textTransform: 'none',
            fontWeight: 500,
            fontSize: '0.9rem',
            color: 'rgba(102, 126, 234, 0.7)',
            '&.Mui-selected': {
              color: '#667eea',
              fontWeight: 600,
            },
            '&:hover': {
              color: '#667eea',
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
            },
          },
          '& .MuiTabs-indicator': {
            backgroundColor: '#667eea',
            height: 3,
            borderRadius: '3px 3px 0 0',
          },
        }}
      >
        {navItems.map((item, index) => (
          <Tab
            key={item.path}
            label={item.label}
            icon={item.icon}
            iconPosition="start"
            component={Link}
            to={item.path}
          />
        ))}
      </Tabs>
    </Box>
  );
}

function AppContent() {
  const { t, language } = useLanguage(); // 同时获取language
  const [activeTask, setActiveTask] = useState(null);
  const [isApiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  
  // 添加语言变化时的调试日志
  useEffect(() => {
    console.log("AppContent检测到语言变化:", language);
    console.log("当前翻译对象:", t);
  }, [language, t]);

  // 安全检查：如果t函数不存在，提供默认值
  const safeT = React.useCallback((key) => {
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
      'nav.apiDocs': 'API 文档',
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
  }, [t, language]); // 添加language作为依赖项，确保语言变化时重新计算

  // Material UI 欢迎页面组件
  const WelcomePage = () => {
    // 直接从上下文获取语言和翻译函数
    const { language, t } = useLanguage();
    
    // 获取屏幕分辨率信息
    React.useEffect(() => {
      // 注释掉屏幕分辨率信息日志
      /*
      console.log('=== 屏幕分辨率信息 ===');
      console.log('屏幕宽度:', window.screen.width);
      console.log('屏幕高度:', window.screen.height);
      console.log('浏览器窗口宽度:', window.innerWidth);
      console.log('浏览器窗口高度:', window.innerHeight);
      console.log('设备像素比:', window.devicePixelRatio);
      console.log('可用屏幕宽度:', window.screen.availWidth);
      console.log('可用屏幕高度:', window.screen.availHeight);
      
      // 监听窗口大小变化
      const handleResize = () => {
        console.log('窗口大小变化 - 宽度:', window.innerWidth, '高度:', window.innerHeight);
      };
      
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
      */
    }, []);

    // 使用语言作为依赖项，确保语言变化时重新渲染
    const features = React.useMemo(() => [
      {
        icon: '📊',
        title: t?.home?.features?.prediction?.title || '智能预测',
        desc: t?.home?.features?.prediction?.desc || '多种算法的能耗预测模型',
        color: 'primary'
      },
      {
        icon: '🔍',
        title: t?.home?.features?.anomaly?.title || '异常检测',
        desc: t?.home?.features?.anomaly?.desc || '实时识别能耗异常模式',
        color: 'secondary'
      },
      {
        icon: '⚡',
        title: t?.home?.features?.async?.title || '异步处理',
        desc: t?.home?.features?.async?.desc || '高效的后台任务处理',
        color: 'info'
      },
      {
        icon: '🎯',
        title: t?.home?.features?.management?.title || '资产管理',
        desc: t?.home?.features?.management?.desc || '统一的设备和资产管理',
        color: 'success'
      },
      {
        icon: '📈',
        title: t?.home?.features?.visualization?.title || '数据可视化',
        desc: t?.home?.features?.visualization?.desc || '直观的图表和报告',
        color: 'warning'
      },
      {
        icon: '🔒',
        title: t?.home?.features?.security?.title || '安全认证',
        desc: t?.home?.features?.security?.desc || 'API密钥和权限管理',
        color: 'error'
      }
    ], [t, language]); // 添加language和t作为依赖项

    return (
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography 
            variant="h1" 
            component="h1" 
            gutterBottom
            sx={{ 
              color: 'white !important',
              fontWeight: 700,
              mb: 2,
              fontSize: { xs: '2.5rem', sm: '3rem', md: '3.5rem' },
              background: 'none !important',
              WebkitBackgroundClip: 'unset !important',
              WebkitTextFillColor: 'white !important',
              backgroundClip: 'unset !important'
            }}
          >
            {t?.home?.title || '能耗预测与异常检测管理系统'}
          </Typography>
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 4, 
              maxWidth: 600, 
              mx: 'auto',
              color: 'rgba(255, 255, 255, 0.95) !important',
              fontWeight: 400,
              fontSize: { xs: '1rem', sm: '1.1rem', md: '1.25rem' }
            }}
          >
            {t?.home?.subtitle || '基于机器学习的工业能耗智能分析平台'}
          </Typography>
          
          <Box sx={{ 
            maxWidth: 1000, 
            mx: 'auto',
            px: 2
          }}>
            {/* 第一行：3个卡片 */}
            <Grid container spacing={3} sx={{ mb: 3, justifyContent: 'center' }}>
              {features.slice(0, 3).map((feature, index) => (
                <Grid 
                  item 
                  xs={12} 
                  sm={6} 
                  md={4}
                  key={index}
                  sx={{
                    display: 'flex',
                    justifyContent: 'center'
                  }}
                >
                  <Card sx={{ 
                    width: 280,
                    height: 200,
                    display: 'flex', 
                    flexDirection: 'column'
                  }}>
                    <CardContent sx={{ 
                      flexGrow: 1, 
                      textAlign: 'center', 
                      p: 3,
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center'
                    }}>
                      <Box sx={{ fontSize: '3rem', mb: 1.5 }}>
                        {feature.icon}
                      </Box>
                      <Typography 
                        variant="h6" 
                        component="h3" 
                        gutterBottom 
                        color="text.primary"
                        sx={{ 
                          fontSize: '1.2rem',
                          fontWeight: 600,
                          mb: 1,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {feature.title}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color="text.secondary" 
                        sx={{ 
                          lineHeight: 1.4,
                          fontSize: '0.9rem',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {feature.desc}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            
            {/* 第二行：3个卡片 */}
            <Grid container spacing={3} sx={{ justifyContent: 'center' }}>
              {features.slice(3, 6).map((feature, index) => (
                <Grid 
                  item 
                  xs={12} 
                  sm={6} 
                  md={4}
                  key={index + 3}
                  sx={{
                    display: 'flex',
                    justifyContent: 'center'
                  }}
                >
                  <Card sx={{ 
                    width: 280,
                    height: 200,
                    display: 'flex', 
                    flexDirection: 'column'
                  }}>
                    <CardContent sx={{ 
                      flexGrow: 1, 
                      textAlign: 'center', 
                      p: 3,
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center'
                    }}>
                      <Box sx={{ fontSize: '3rem', mb: 1.5 }}>
                        {feature.icon}
                      </Box>
                      <Typography 
                        variant="h6" 
                        component="h3" 
                        gutterBottom 
                        color="text.primary"
                        sx={{ 
                          fontSize: '1.2rem',
                          fontWeight: 600,
                          mb: 1,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {feature.title}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color="text.secondary" 
                        sx={{ 
                          lineHeight: 1.4,
                          fontSize: '0.9rem',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {feature.desc}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Box>
      </Container>
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
      
      <Box sx={{ 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <AppBar 
          position="sticky" 
          elevation={0}
          sx={{ 
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
            color: 'text.primary'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', p: 1, position: 'relative' }}>
            <Box sx={{ flexGrow: 1 }}>
              <Navigation />
            </Box>
          </Box>
        </AppBar>
        
        {/* 将语言切换按钮放在固定位置 */}
        <Box sx={{ 
          position: 'fixed', 
          top: 10, 
          right: 16, 
          zIndex: 2000
        }}>
          <LanguageToggle />
        </Box>

        <Box component="main" sx={{ flexGrow: 1, py: 3 }}>
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/forecast" element={
              <Container maxWidth="xl">
                <Card>
                  <CardContent>
                    <ForecastView />
                  </CardContent>
                </Card>
              </Container>
            } />
            <Route path="/anomaly-detection" element={
              <Container maxWidth="xl">
                <Card>
                  <CardContent>
                    <AnomalyDetectionView />
                  </CardContent>
                </Card>
              </Container>
            } />
            <Route path="/training" element={
              <Container maxWidth="xl">
                <Card>
                  <CardContent>
                    <ModelTraining 
                      setActiveTask={setActiveTask} 
                      activeTask={activeTask} 
                    />
                  </CardContent>
                </Card>
              </Container>
            } />
            <Route path="/assets" element={
              <Container maxWidth="xl">
                <AssetList />
              </Container>
            } />
            <Route path="/models" element={
              <Container maxWidth="xl">
                <ModelList />
              </Container>
            } />
            <Route path="/api-keys" element={
              <Container maxWidth="xl">
                <ApiKeyList />
              </Container>
            } />
            <Route path="/api-docs" element={
              <Container maxWidth="xl">
                <Card>
                  <CardContent>
                    <ApiDocumentation />
                  </CardContent>
                </Card>
              </Container>
            } />
          </Routes>
          
          {activeTask && (
            <Box sx={{
              position: 'fixed',
              bottom: 16,
              right: 16,
              zIndex: 1000,
              minWidth: 350
            }}>
              <Card elevation={8}>
                <CardContent>
                  <TaskStatus task={activeTask} onTaskComplete={handleTaskComplete} />
                </CardContent>
              </Card>
            </Box>
          )}
        </Box>
      </Box>
    </Router>
  );
}

function App() {
  // 获取当前语言，用于强制整个应用在语言切换时重新渲染
  const [currentLanguage, setCurrentLanguage] = useState(() => {
    return localStorage.getItem('language') || 'zh';
  });
  
  // 监听localStorage的变化，当语言改变时更新state
  useEffect(() => {
    const handleStorageChange = () => {
      const newLanguage = localStorage.getItem('language') || 'zh';
      if (newLanguage !== currentLanguage) {
        console.log("App检测到语言变化:", newLanguage);
        setCurrentLanguage(newLanguage);
      }
    };
    
    // 添加事件监听器
    window.addEventListener('storage', handleStorageChange);
    
    // 移除定时器，只依赖storage事件和自定义事件
    const handleLanguageChange = (event) => {
      const newLanguage = event.detail || localStorage.getItem('language') || 'zh';
      if (newLanguage !== currentLanguage) {
        console.log("App检测到语言变化:", newLanguage);
        setCurrentLanguage(newLanguage);
      }
    };
    
    window.addEventListener('languageChanged', handleLanguageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('languageChanged', handleLanguageChange);
    };
  }, [currentLanguage]);
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LanguageProvider>
        <AppContent />
      </LanguageProvider>
    </ThemeProvider>
  );
}

export default App;
