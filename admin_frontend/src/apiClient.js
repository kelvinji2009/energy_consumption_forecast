const API_KEY_STORAGE_KEY = 'app-api-key';

/**
 * A wrapper around the native fetch API that automatically adds the 
 * Authorization header and handles common API request logic.
 *
 * @param {string} endpoint The API endpoint to call (e.g., '/admin/assets').
 * @param {object} options The options object for the fetch call (e.g., method, body, headers).
 * @returns {Promise<any>} A promise that resolves to the JSON response.
 * @throws {Error} Throws an error if the network response is not ok.
 */
const apiClient = async (endpoint, options = {}) => {
  const apiKey = localStorage.getItem(API_KEY_STORAGE_KEY);

  if (!apiKey) {
    // This should ideally not happen if the UI prevents calls without a key,
    // but it's a good safeguard.
    return Promise.reject(new Error('API Key not found in storage.'));
  }

  let headers = {
    'Authorization': `Bearer ${apiKey}`,
  };

  // If the body is not FormData, set Content-Type to application/json
  // Otherwise, the browser will automatically set multipart/form-data with the correct boundary
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const config = {
    ...options,
    headers: {
      ...headers,
      ...options.headers, // Allow overriding headers from options
    },
  };

  const response = await fetch(`/api${endpoint}`, config);

  if (!response.ok) {
    // Try to parse the error body for a more specific message
    const errorData = await response.json().catch(() => null);
    let errorMessage = errorData?.detail || `HTTP error! status: ${response.status}`;
    if (typeof errorMessage === 'object') {
      errorMessage = JSON.stringify(errorMessage, null, 2);
    }
    throw new Error(errorMessage);
  }

  // If the response has no content (e.g., for a 204 response)
  if (response.status === 204) {
    return null;
  }

  return response.json();
};

export default apiClient;
