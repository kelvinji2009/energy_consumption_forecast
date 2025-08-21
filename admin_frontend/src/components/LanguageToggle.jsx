import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

function LanguageToggle() {
  const { language, toggleLanguage } = useLanguage();

  return (
    <button
      onClick={toggleLanguage}
      style={{
        background: 'rgba(255, 255, 255, 0.9)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        borderRadius: '25px',
        padding: '0.5rem 1rem',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.9rem',
        fontWeight: '500',
        color: '#4a5568',
        transition: 'all 0.3s ease',
        backdropFilter: 'blur(5px)',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
      }}
      onMouseEnter={(e) => {
        e.target.style.background = 'rgba(255, 255, 255, 1)';
        e.target.style.transform = 'translateY(-1px)';
        e.target.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
      }}
      onMouseLeave={(e) => {
        e.target.style.background = 'rgba(255, 255, 255, 0.9)';
        e.target.style.transform = 'translateY(0)';
        e.target.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
      }}
    >
      <span style={{ fontSize: '1.2rem' }}>
        {language === 'zh' ? '🇨🇳' : '🇺🇸'}
      </span>
      <span>
        {language === 'zh' ? '中文' : 'English'}
      </span>
    </button>
  );
}

export default LanguageToggle;