import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../apiClient';
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

function ForecastView() {
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [models, setModels] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [forecastHorizon, setForecastHorizon] = useState(168); // Default to 168 hours (1 week)
    const [dataInputMethod, setDataInputMethod] = useState('upload'); // 'upload' or 's3'
    const [s3DataPath, setS3DataPath] = useState('');
    const [forecastResult, setForecastResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [chartData, setChartData] = useState({ datasets: [] });
    const [selectedFile, setSelectedFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const [maxForecastHorizon, setMaxForecastHorizon] = useState(null);
    const chartRef = useRef(null);

    // Fetch assets on component mount
    useEffect(() => {
        const fetchAssets = async () => {
            try {
                const data = await apiClient('/admin/assets');
                setAssets(data);
                if (data.length > 0) {
                    setSelectedAsset(prev => prev || data[0].id); // 只在未选中时赋值
                }
            } catch (err) {
                console.error("Failed to fetch assets:", err);
                setError("Could not load assets. Is the API server running and the API key correct?");
            }
        };
        fetchAssets();
    }, []);

    // Fetch models when selectedAsset changes
    useEffect(() => {
        if (selectedAsset) {
            setLoading(true);
            setError(null);
            const fetchModels = async () => {
                try {
                    const data = await apiClient(`/admin/models?asset_id=${selectedAsset}&status=COMPLETED`);
                    // Sort models by creation date (newest first)
                    const sortedModels = data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                    setModels(sortedModels);
                    if (sortedModels.length > 0) {
                        setSelectedModelId(prev => prev || sortedModels[0].id); // 只在未选中时赋值
                    }
                } catch (err) {
                    console.error("Failed to fetch models:", err);
                    setError("Could not load models for the selected asset.");
                } finally {
                    setLoading(false);
                }
            };
            fetchModels();
        }
    }, [selectedAsset]);

    useEffect(() => {
        if (maxForecastHorizon !== null && forecastHorizon > maxForecastHorizon) {
            setForecastHorizon(Math.max(1, maxForecastHorizon));
        }
    }, [maxForecastHorizon, forecastHorizon]);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type === "text/csv") {
            setSelectedFile(file);
            setFileName(file.name);
            setError(null);

            // --- Instant Feedback Logic ---
            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.error("CSV parsing errors:", results.errors);
                        setError(`CSV 解析错误: ${results.errors[0].message}`);
                        return;
                    }
                    const nonEmptyRows = results.data.filter(row => Object.values(row).some(val => val !== null && val !== ''));
                    const historical_hours = nonEmptyRows.length;
                    const max_horizon = Math.floor(historical_hours / 4);
                    setMaxForecastHorizon(max_horizon);

                    if (max_horizon <= 0) {
                        setError('CSV 数据量太少，无法预测，请上传更多历史数据。');
                        return;
                    }
                },
                error: (err) => {
                    console.error("PapaParse error:", err);
                    setError(`CSV 文件读取失败: ${err.message}`);
                }
            });

        } else {
            setSelectedFile(null);
            setFileName('');
            setMaxForecastHorizon(null);
            setError('Please select a valid .csv file.');
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setForecastResult(null);

        if (!assets.length) {
            setError("资产列表未加载，请稍后重试。");
            setLoading(false);
            return;
        }
        if (!selectedAsset) {
            setError("请选择资产。");
            setLoading(false);
            return;
        }
        if (!models.length) {
            setError("模型列表未加载，请稍后重试。");
            setLoading(false);
            return;
        }
        if (!selectedModelId) {
            setError("请选择模型。");
            setLoading(false);
            return;
        }
        if (!forecastHorizon) {
            setError("请输入预测步长。");
            setLoading(false);
            return;
        }

        let url = '';
        const options = { method: 'POST' };

        if (dataInputMethod === 'upload') {
            if (!selectedFile) {
                setError("Please select a CSV file to upload.");
                setLoading(false);
                return;
            }
            const formData = new FormData();
            formData.append('forecast_horizon', forecastHorizon);
            formData.append('model_id', selectedModelId);
            formData.append('file', selectedFile);
            options.body = formData;
            url = `/assets/${selectedAsset}/predict_from_csv`;

        } else if (dataInputMethod === 's3') {
            if (!s3DataPath) {
                setError("Please enter an S3 data path.");
                setLoading(false);
                return;
            }
            const params = new URLSearchParams({
                s3_data_path: s3DataPath,
                forecast_horizon: forecastHorizon,
                model_id: selectedModelId,
            });
            url = `/assets/${selectedAsset}/predict_from_s3?${params.toString()}`;
        }

        try {
            const result = await apiClient(url, options);

            const processAndSetChartData = (historical, forecast) => {
                const historicalData = historical
                    .filter(d => d.timestamp && d.value != null)
                    .map(d => ({ x: d.timestamp, y: d.value }));

                const forecastData = forecast.map(d => ({ x: d.timestamp, y: d.predicted_value }));

                setChartData({
                    datasets: [
                        {
                            label: 'Historical Energy',
                            data: historicalData,
                            borderColor: '#8884d8',
                            backgroundColor: '#8884d8',
                            pointRadius: 0,
                        },
                        {
                            label: 'Forecasted Energy',
                            data: forecastData,
                            borderColor: '#82ca9d',
                            backgroundColor: '#82ca9d',
                            pointRadius: 0,
                        }
                    ]
                });
            };

            // Use historical data from API response if available
            if (result.historical_data && result.historical_data.length > 0) {
                processAndSetChartData(result.historical_data, result.forecast_data);
            } else if (dataInputMethod === 'upload' && selectedFile) {
                // Fallback for CSV upload if backend doesn't return historical data
                Papa.parse(selectedFile, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (parsedResult) => {
                        if (parsedResult.errors.length) {
                            setError(`CSV Parsing Error: ${parsedResult.errors[0].message}`);
                            return;
                        }
                        const apiHistorical = parsedResult.data.map(d => ({ ...d, value: d.value ?? d.energy_kwh }));
                        processAndSetChartData(apiHistorical, result.forecast_data);
                    },
                    error: (err) => setError(`CSV Parsing Error: ${err.message}`),
                });
            } else {
                // If no historical data is available at all, just plot the forecast
                processAndSetChartData([], result.forecast_data);
            }

        } catch (err) {
            console.error("Prediction error:", err);
            setError(`Prediction failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    // chartOptions 定义移到 return 之前
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
        animation: false,
    };

    return (
        <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
            <h2>能耗预测</h2>

            <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="asset-select" style={{ display: 'block', marginBottom: '0.5rem' }}>选择资产:</label>
                <select
                    id="asset-select"
                    value={selectedAsset}
                    onChange={e => setSelectedAsset(e.target.value)}
                    style={{ width: '100%', padding: '0.5rem' }}
                >
                    {assets.map(asset => (
                        <option key={asset.id} value={asset.id}>{asset.name} ({asset.id})</option>
                    ))}
                </select>
            </div>

            <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="model-select" style={{ display: 'block', marginBottom: '0.5rem' }}>选择模型:</label>
                <select
                    id="model-select"
                    value={selectedModelId}
                    onChange={e => setSelectedModelId(e.target.value)}
                    style={{ width: '100%', padding: '0.5rem' }}
                    disabled={!selectedAsset || models.length === 0}
                >
                    {models.length === 0 ? (
                        <option value="">No completed models available</option>
                    ) : (
                        models
                            .filter(model => model.status === 'COMPLETED')
                            .map(model => (
                            <option key={model.id} value={model.id}>
                                {`v${model.model_version} - ${model.model_type} | MAPE: ${model.metrics?.mape?.toFixed(2) ?? 'N/A'}% | Trained: ${new Date(model.created_at).toLocaleDateString()}`}
                            </option>
                        ))
                    )}
                </select>
            </div>

            <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="forecast-horizon" style={{ display: 'block', marginBottom: '0.5rem' }}>预测步长 (小时):</label>
                <input
                    id="forecast-horizon"
                    type="number"
                    value={forecastHorizon}
                    onChange={e => setForecastHorizon(Math.max(1, parseInt(e.target.value, 10)))}
                    min="1"
                    max={maxForecastHorizon || undefined}
                    style={{ width: '100%', padding: '0.5rem' }}
                />
            </div>

            <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem' }}>数据输入方式:</label>
                <input
                    type="radio"
                    id="upload-radio"
                    name="dataInputMethod"
                    value="upload"
                    checked={dataInputMethod === 'upload'}
                    onChange={() => setDataInputMethod('upload')}
                />
                <label htmlFor="upload-radio" style={{ marginRight: '1rem' }}>上传 CSV 文件</label>
                <input
                    type="radio"
                    id="s3-radio"
                    name="dataInputMethod"
                    value="s3"
                    checked={dataInputMethod === 's3'}
                    onChange={() => setDataInputMethod('s3')}
                />
                <label htmlFor="s3-radio">S3 路径</label>
            </div>

            {dataInputMethod === 'upload' && (
                <div style={{ marginBottom: '1rem' }}>
                    <label htmlFor="csv-upload" style={{ display: 'block', marginBottom: '0.5rem' }}>上传历史数据 CSV:</label>
                    <input
                        id="csv-upload"
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        style={{ width: '100%', padding: '0.5rem' }}
                    />
                </div>
            )}

            {dataInputMethod === 's3' && (
                <div style={{ marginBottom: '1rem' }}>
                    <label htmlFor="s3-data-path" style={{ display: 'block', marginBottom: '0.5rem' }}>S3 数据路径 (Key):</label>
                    <input
                        id="s3-data-path"
                        type="text"
                        value={s3DataPath}
                        onChange={e => setS3DataPath(e.target.value)}
                        placeholder="e.g., historical-data/asset_a_history.csv"
                        style={{ width: '100%', padding: '0.5rem' }}
                    />
                </div>
            )}

            <button onClick={handlePredict}
                disabled={loading || !assets.length || !models.length}
                style={{ padding: '0.75rem', cursor: 'pointer' }}>
                {loading ? '预测中...' : '开始预测'}
            </button>

            {error && <p style={{ color: 'red', marginTop: '1rem' }}>错误: {error}</p>}

            <div style={{ position: 'relative', width: '100%', height: '400px', marginTop: '20px' }}>
                <Line ref={chartRef} options={chartOptions} data={chartData} />
            </div>
        </div>
    );
}

export default ForecastView;
