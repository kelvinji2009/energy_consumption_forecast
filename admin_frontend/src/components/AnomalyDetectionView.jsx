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

function AnomalyDetectionView() {
    const [assets, setAssets] = useState([]);
    const [selectedAsset, setSelectedAsset] = useState('');
    const [models, setModels] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [dataInputMethod, setDataInputMethod] = useState('upload'); // 'upload' or 's3'
    const [s3DataPath, setS3DataPath] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [chartData, setChartData] = useState({ datasets: [] });
    const [selectedFile, setSelectedFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const chartRef = useRef(null);

    // Fetch assets on component mount
    useEffect(() => {
        const fetchAssets = async () => {
            try {
                const data = await apiClient('/admin/assets');
                setAssets(data);
                if (data.length > 0) {
                    setSelectedAsset(prev => prev || data[0].id);
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
            setModels([]); // Clear previous models
            setSelectedModelId(''); // Reset selection
            const fetchModels = async () => {
                try {
                    const data = await apiClient(`/admin/models?asset_id=${selectedAsset}`);
                    const modelsWithDetectors = data
                        .filter(m => m.status === 'COMPLETED' && m.detector_path);
                    const sortedModels = modelsWithDetectors.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                    
                    setModels(sortedModels);
                    if (sortedModels.length > 0) {
                        // Always set to the newest model of the *new* asset
                        setSelectedModelId(sortedModels[0].id);
                    } else {
                        setError("No models with anomaly detectors found for this asset.");
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

    const handleDetect = async () => {
        setLoading(true);
        setError(null);
        setChartData({ datasets: [] });

        if (!selectedAsset || !selectedModelId) {
            setError("Please select an asset and a model.");
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
            formData.append('model_id', selectedModelId);
            formData.append('file', selectedFile);
            options.body = formData;
            url = `/assets/${selectedAsset}/detect_anomalies_from_csv`;

        } else if (dataInputMethod === 's3') {
            if (!s3DataPath) {
                setError("Please enter an S3 data path.");
                setLoading(false);
                return;
            }
            const params = new URLSearchParams({
                s3_data_path: s3DataPath,
                model_id: selectedModelId,
            });
            url = `/assets/${selectedAsset}/detect_anomalies_from_s3?${params.toString()}`;
        }

        try {
            const result = await apiClient(url, options);

            const historicalData = result.historical_data
                .filter(d => d.timestamp && d.value != null)
                .map(d => ({ x: new Date(d.timestamp), y: d.value }));

            const anomalyPoints = result.anomalies.map(a => ({
                x: new Date(a.timestamp),
                y: a.value
            }));

            setChartData({
                datasets: [
                    {
                        label: 'Historical Energy',
                        data: historicalData,
                        borderColor: '#8884d8',
                        backgroundColor: 'rgba(136, 132, 216, 0.5)',
                        pointRadius: 1,
                        type: 'line',
                    },
                    {
                        label: 'Anomalies',
                        data: anomalyPoints,
                        backgroundColor: 'red',
                        pointRadius: 5,
                        type: 'scatter',
                    }
                ]
            });

        } catch (err) {
            console.error("Detection error:", err);
            setError(`Detection failed: ${err.message}`);
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
                text: 'Anomaly Detection Results'
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
            <h2>异常检测</h2>

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
                        <option value="">No models with detectors available</option>
                    ) : (
                        models.map(model => (
                            <option key={model.id} value={model.id}>
                                {`v${model.model_version} - ${model.model_type} | MAPE: ${model.metrics?.mape?.toFixed(2) ?? 'N/A'}% | Trained: ${new Date(model.created_at).toLocaleDateString()} (ID: ${model.id})`}
                            </option>
                        ))
                    )}
                </select>
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

            <button onClick={handleDetect}
                disabled={loading || !assets.length || !models.length}
                style={{ padding: '0.75rem', cursor: 'pointer' }}>
                {loading ? '检测中...' : '开始检测'}
            </button>

            {error && <p style={{ color: 'red', marginTop: '1rem' }}>错误: {error}</p>}

            <div style={{ position: 'relative', width: '100%', height: '400px', marginTop: '20px' }}>
                <Line ref={chartRef} options={chartOptions} data={chartData} />
            </div>
        </div>
    );
}

export default AnomalyDetectionView;
