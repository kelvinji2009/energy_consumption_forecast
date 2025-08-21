const API_PREFIX = '/api'; // All API calls will be prefixed with /api to be caught by the Nginx proxy.
const API_KEY_STORAGE_KEY = 'ENERGY_FORECAST_API_KEY';

// This singleton promise prevents multiple modals from appearing at once.
let apiKeyPromise = null;

/**
 * Retrieves the API key from localStorage or prompts the user for it via a global event.
 * @returns {Promise<string>} A promise that resolves with the API key.
 */
const getApiKey = () => {
  const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
  if (storedKey) {
    return Promise.resolve(storedKey);
  }

  // If a prompt is already active, return the existing promise to avoid duplicates.
  if (apiKeyPromise) {
    return apiKeyPromise;
  }

  // Create a new promise that will resolve once the 'api-key-provided' event is dispatched.
  apiKeyPromise = new Promise((resolve) => {
    const handleKeyProvided = (event) => {
      const newKey = event.detail;
      if (newKey) {
        localStorage.setItem(API_KEY_STORAGE_KEY, newKey);
        window.removeEventListener('api-key-provided', handleKeyProvided);
        apiKeyPromise = null; // Reset for the next time a key is needed.
        resolve(newKey);
      }
    };

    window.addEventListener('api-key-provided', handleKeyProvided, { once: true });
    // Dispatch an event to notify the UI that a key is required.
    window.dispatchEvent(new CustomEvent('request-api-key'));
  });

  return apiKeyPromise;
};

const apiClient = async (endpoint, options = {}) => {
  const apiKey = await getApiKey();

  const headers = {
    'X-API-Key': apiKey,
    ...options.headers,
  };

  // If the body is FormData, let the browser set the Content-Type header automatically.
  // Otherwise, set it to application/json.
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  // Construct the final URL with the /api prefix
  const response = await fetch(`${API_PREFIX}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    // If the key is invalid (401/403), remove it. The next API call will trigger a new prompt.
    if (response.status === 401 || response.status === 403) {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
    const errorData = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(errorData.detail || errorData.message || 'An unknown error occurred');
  }

  if (response.status === 204) { // No Content
    return null;
  }

  return response.json();
};

export default apiClient;
