import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';
import Papa from 'papaparse';

// Register Chart.js components and plugins
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  zoomPlugin
);

const API_BASE_URL = 'http://localhost:8000';

function ForecastView() {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [chartData, setChartData] = useState({ datasets: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const chartRef = useRef(null);

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/assets`);
        if (!response.ok) throw new Error('Failed to fetch assets');
        const data = await response.json();
        setAssets(data);
        if (data.length > 0) setSelectedAsset(data[0].id);
      } catch (err) {
        setError(err.message);
      }
    };
    fetchAssets();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "text/csv") {
      setSelectedFile(file);
      setFileName(file.name);
      setError(null);
    } else {
      setSelectedFile(null);
      setFileName('');
      setError('Please select a valid .csv file.');
    }
  };

  const handleForecast = async () => {
    if (!selectedAsset || !selectedFile) {
      setError('Please select an asset and a CSV file.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const tempApiKey = "3369df94-7513-459e-be83-104bdb046b85";
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE_URL}/assets/${selectedAsset}/predict_from_csv`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${tempApiKey}` },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction API call failed');
      }

      const result = await response.json();

      Papa.parse(selectedFile, {
        header: true,
        dynamicTyping: true,
        complete: (parsedResult) => {
          const historicalData = parsedResult.data
            .filter(d => d.timestamp && (d.value != null || d.energy_kwh != null))
            .map(d => ({ x: d.timestamp, y: d.value != null ? d.value : d.energy_kwh }));

          const forecastData = result.forecast_data.map(d => ({ x: d.timestamp, y: d.predicted_value }));

          setChartData({
            datasets: [
              {
                label: 'Historical Energy',
                data: historicalData,
                borderColor: '#8884d8',
                backgroundColor: '#8884d8',
                pointRadius: 0, // Hide points for performance
              },
              {
                label: 'Forecasted Energy',
                data: forecastData,
                borderColor: '#82ca9d',
                backgroundColor: '#82ca9d',
                pointRadius: 0, // Hide points for performance
              }
            ]
          });
        },
        error: (err) => setError(`CSV Parsing Error: ${err.message}`),
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          tooltipFormat: 'MMM dd, yyyy HH:mm',
        },
        title: {
          display: true,
          text: 'Timestamp'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Energy (kWh)'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Energy Consumption Forecast'
      },
      zoom: {
        pan: {
          enabled: true,
          mode: 'x',
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
        }
      }
    },
    animation: false, // Disable animation for performance
  };

  return (
    <div className="card">
      <h2>Energy Consumption Forecast</h2>
      <div className="controls" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <select value={selectedAsset} onChange={e => setSelectedAsset(e.target.value)} disabled={loading}>
          <option value="">Select an Asset</option>
          {assets.map(asset => <option key={asset.id} value={asset.id}>{asset.name}</option>)}
        </select>
        
        <input 
          type="file" 
          id="csv-upload" 
          accept=".csv"
          onChange={handleFileChange} 
          style={{ display: 'none' }}
          disabled={loading}
        />
        <label htmlFor="csv-upload" className="button" style={{ cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.6 : 1 }}>
          {fileName || 'Select CSV File'}
        </label>

        <button onClick={handleForecast} disabled={loading || !selectedFile}>
          {loading ? 'Generating...' : 'Start Forecast'}
        </button>
        <button onClick={() => chartRef.current?.resetZoom()} disabled={loading}>
          Reset Zoom
        </button>
      </div>

      {error && <p className="error">Error: {error}</p>}

      <div style={{ position: 'relative', width: '100%', height: '400px', marginTop: '20px' }}>
        <Line ref={chartRef} options={chartOptions} data={chartData} />
      </div>
    </div>
  );
}

export default ForecastView;