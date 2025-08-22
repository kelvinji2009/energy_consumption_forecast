import React from 'react';
import { Button, Box } from '@mui/material';
import { useLanguage } from '../contexts/LanguageContext';

function LanguageToggle() {
  const { language, toggleLanguage } = useLanguage();
  
  console.log("LanguageToggle渲染，当前语言:", language); // 添加调试日志
  
  const handleClick = (e) => {
    e.stopPropagation(); // 阻止事件冒泡
    console.log("语言切换按钮被点击，当前语言:", language); // 添加调试日志
    toggleLanguage();
    console.log("toggleLanguage函数已调用");
    
    // 触发自定义事件通知App组件语言已变化
    setTimeout(() => {
      const newLanguage = localStorage.getItem('language') || 'zh';
      window.dispatchEvent(new CustomEvent('languageChanged', { detail: newLanguage }));
    }, 100);
  };

  return (
    <Button
      onClick={handleClick}
      variant="contained"
      disableElevation
      sx={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '25px',
        padding: '0.5rem 1rem',
        textTransform: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.9rem',
        fontWeight: '500',
        color: 'white',
        transition: 'all 0.3s ease',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        zIndex: 2000,
        '&:hover': {
          background: 'linear-gradient(135deg, #5a6fd9 0%, #6a3e99 100%)',
          transform: 'translateY(-1px)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        }
      }}
    >
      <Box component="span" sx={{ fontSize: '1.2rem', mr: 0.5 }}>
        {language === 'zh' ? '🇨🇳' : '🇺🇸'}
      </Box>
      <Box component="span">
        {language === 'zh' ? '中文' : 'English'}
      </Box>
    </Button>
  );
}

export default LanguageToggle;