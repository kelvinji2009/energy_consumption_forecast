import React, { useRef } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

function CustomFileInput({ 
  onFileChange, 
  accept = ".csv", 
  selectedFile = null, 
  disabled = false,
  required = false,
  style = {}
}) {
  const { t, language } = useLanguage();
  const fileInputRef = useRef(null);

  // 翻译函数
  const getText = (key) => {
    if (!t || typeof t !== 'object') {
      return key;
    }
    
    const keys = key.split('.');
    let result = t;
    for (const k of keys) {
      if (result && typeof result === 'object' && k in result) {
        result = result[k];
      } else {
        return key;
      }
    }
    
    return result || key;
  };

  const handleButtonClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (onFileChange) {
      onFileChange(file);
    }
  };

  return (
    <div style={{ width: '100%', ...style }}>
      {/* 隐藏的原生文件输入 */}
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        required={required}
        disabled={disabled}
        style={{ display: 'none' }}
      />
      
      {/* 自定义按钮 */}
      <button
        type="button"
        onClick={handleButtonClick}
        disabled={disabled}
        style={{
          width: '100%',
          padding: '1rem',
          border: '2px dashed rgba(102, 126, 234, 0.3)',
          background: disabled ? '#f7fafc' : 'rgba(102, 126, 234, 0.05)',
          borderRadius: '10px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          fontSize: '1rem',
          color: disabled ? '#a0aec0' : '#4a5568',
          transition: 'all 0.3s ease',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem'
        }}
        onMouseEnter={(e) => {
          if (!disabled) {
            e.target.style.background = 'rgba(102, 126, 234, 0.1)';
            e.target.style.borderColor = 'rgba(102, 126, 234, 0.5)';
          }
        }}
        onMouseLeave={(e) => {
          if (!disabled) {
            e.target.style.background = 'rgba(102, 126, 234, 0.05)';
            e.target.style.borderColor = 'rgba(102, 126, 234, 0.3)';
          }
        }}
      >
        📁 {getText('common.chooseFile')}
      </button>
      
      {/* 显示选中的文件 */}
      {selectedFile && (
        <div style={{
          marginTop: '0.5rem',
          padding: '0.5rem 1rem',
          background: 'rgba(72, 187, 120, 0.1)',
          border: '1px solid rgba(72, 187, 120, 0.2)',
          borderRadius: '8px',
          color: '#48bb78',
          fontSize: '0.9rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          ✅ {getText('common.fileSelected')}: {selectedFile.name}
        </div>
      )}
    </div>
  );
}

export default CustomFileInput;