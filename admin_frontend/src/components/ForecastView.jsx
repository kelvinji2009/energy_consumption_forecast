import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush } from 'recharts';
import Papa from 'papaparse'; // Using Papa Parse for robust CSV parsing

const API_BASE_URL = 'http://localhost:8000';

function ForecastView() {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');

  // Fetch assets on component mount
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
    if (!selectedAsset) {
      setError('Please select an asset.');
      return;
    }
    if (!selectedFile) {
      setError('Please select a CSV file to upload.');
      return;
    }

    setLoading(true);
    setError(null);
    setChartData([]);

    try {
      const tempApiKey = "3369df94-7513-459e-be83-104bdb046b85"; // Replace with a valid key if needed
      
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE_URL}/assets/${selectedAsset}/predict_from_csv` , {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${tempApiKey}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction API call failed');
      }

      const result = await response.json();

      // Parse the uploaded CSV to get historical data for the chart
      Papa.parse(selectedFile, {
        header: true,
        dynamicTyping: true,
        complete: (parsedResult) => {
            const historicalData = parsedResult.data;
            
            const historicalChartData = historicalData
                .filter(d => d.timestamp && (d.value != null || d.energy_kwh != null))
                .map(d => ({
                    timestamp: d.timestamp,
                    'Historical Energy': d.value != null ? d.value : d.energy_kwh,
                }));

            const forecastChartData = result.forecast_data.map(d => ({
                timestamp: d.timestamp,
                'Forecasted Energy': d.predicted_value,
            }));

            const allData = [...historicalChartData, ...forecastChartData]
                .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

            setChartData(allData);
        },
        error: (err) => {
            setError(`CSV Parsing Error: ${err.message}`);
        }
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatXAxis = (tickItem) => {
    return new Date(tickItem).toLocaleString([], { month: 'numeric', day: 'numeric', hour: '2-digit' });
  };

  return (
    <div className="card">
      <h2>Energy Consumption Forecast</h2>
      <div className="controls" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <select value={selectedAsset} onChange={e => setSelectedAsset(e.target.value)} disabled={loading}>
          <option value="">Select an Asset</option>
          {assets.map(asset => (
            <option key={asset.id} value={asset.id}>{asset.name}</option>
          ))}
        </select>
        
        <input 
          type="file" 
          id="csv-upload" 
          accept=".csv"
          onChange={handleFileChange} 
          style={{ display: 'none' }} // Hide the default input
          disabled={loading}
        />
        <label htmlFor="csv-upload" className="button" style={{ cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.6 : 1 }}>
          {fileName || 'Select CSV File'}
        </label>

        <button onClick={handleForecast} disabled={loading || !selectedFile}>
          {loading ? 'Generating...' : 'Start Forecast'}
        </button>
      </div>

      {error && <p className="error">Error: {error}</p>}

      <div style={{ width: '100%', height: 400, marginTop: '20px' }}>
        <ResponsiveContainer>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              angle={-30} 
              textAnchor="end" 
              height={70}
              tickFormatter={formatXAxis}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={(label) => new Date(label).toLocaleString()}
            />
            <Legend />
            <Line type="monotone" dataKey="Historical Energy" stroke="#8884d8" dot={false} />
            <Line type="monotone" dataKey="Forecasted Energy" stroke="#82ca9d" strokeDasharray="5 5" />
            <Brush dataKey='timestamp' height={30} stroke="#8884d8"/>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default ForecastView;