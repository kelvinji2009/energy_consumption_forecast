import React, { useState } from 'react';

/**
 * A modal dialog that prompts the user to enter an API key.
 */
function ApiKeyPromptModal({ onKeySubmit }) {
  const [apiKey, setApiKey] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (apiKey.trim()) {
      // Dispatch a global event with the new key for the apiClient to catch.
      window.dispatchEvent(new CustomEvent('api-key-provided', { detail: apiKey.trim() }));
      // Call the parent handler to close the modal.
      onKeySubmit();
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <h2>API Key Required</h2>
        <p>Please enter your API key to continue. You can generate one using the command-line tool if you don't have one.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your API key"
            style={styles.input}
            autoFocus
          />
          <button type="submit" style={styles.button}>Submit Key</button>
        </form>
      </div>
    </div>
  );
}

// Basic styling for the modal.
const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: '#ffffff',
    padding: '25px 40px',
    borderRadius: '8px',
    width: '90%',
    maxWidth: '450px',
    boxShadow: '0 5px 15px rgba(0,0,0,0.3)',
    textAlign: 'center',
  },
  input: {
    width: '100%',
    padding: '12px',
    marginBottom: '20px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    boxSizing: 'border-box',
    fontSize: '16px',
  },
  button: {
    width: '100%',
    padding: '12px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#007bff',
    color: 'white',
    cursor: 'pointer',
    fontSize: '16px',
  },
};

export default ApiKeyPromptModal;
