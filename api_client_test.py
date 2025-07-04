import requests
import json
from datetime import datetime, timedelta

def run_prediction_test(api_key: str):
    """向本地运行的API服务发送一个测试预测请求。"""

    print("\n--- API Client Prediction Test ---")

    asset_id = "production_line_A"
    url = f"http://127.0.0.1:8000/assets/{asset_id}/predict"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    input_chunk_length = 168
    historical_data = []
    start_time = datetime.now() - timedelta(hours=input_chunk_length)
    for i in range(input_chunk_length):
        historical_data.append({
            "timestamp": (start_time + timedelta(hours=i)).isoformat(),
            "value": 100 + i * 0.5 + (-1)**i * 5, # 模拟一些波动
            "temp": 25.0,
            "production": 500 + i * 2
        })

    request_payload = {
        "historical_data": historical_data,
        "forecast_horizon": 12
    }

    print(f"Sending POST request to: {url}")
    print(f"Historical data points sent: {len(historical_data)}")

    try:
        response = requests.post(url, headers=headers, json=request_payload, timeout=20) # 增加超时时间

        print(f"\nResponse Status Code: {response.status_code}")

        if response.status_code == 200:
            print("Prediction successful!")
            print("Response JSON:", json.dumps(response.json(), indent=2))
        else:
            print("Prediction failed.")
            print("Error Response:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while connecting to the API server: {e}")
        print("Please ensure the API server is running correctly.")

def run_anomaly_detection_test(api_key: str):
    """向本地运行的API服务发送一个测试异常检测请求。"""

    print("\n--- API Client Anomaly Detection Test ---")

    asset_id = "production_line_A"
    url = f"http://127.0.0.1:8000/assets/{asset_id}/detect_anomalies"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # --- 核心修复 ---
    # 我们需要发送比模型输入块长度更多的数据点。
    # input_chunk_length (168) 是历史，之后的数据点是需要被检测的。
    # 我们发送 168 + 24 = 192 个点，模拟用过去7天的数据来检测新一天的数据。
    input_chunk_length_for_ad = 168
    points_to_detect = 24 
    total_points = input_chunk_length_for_ad + points_to_detect

    data_stream = []
    start_time = datetime.now() - timedelta(hours=total_points)
    
    for i in range(total_points):
        value = 100 + i * 0.5 + (-1)**i * 2 # 正常波动
        
        # 在要被检测的数据点中（即最后24个点）注入一个异常
        # total_points - 5 是倒数第5个点
        if i == total_points - 5: 
            value = 500 # 异常高值
            
        data_stream.append({
            "timestamp": (start_time + timedelta(hours=i)).isoformat(),
            "value": value,
            "temp": 25.0,
            "production": 500 + i * 2
        })

    request_payload = {
        "data_stream": data_stream
    }

    print(f"Sending POST request to: {url}")
    print(f"Data points sent for anomaly detection: {len(data_stream)}")

    try:
        response = requests.post(url, headers=headers, json=request_payload, timeout=20) # 增加超时时间

        print(f"\nResponse Status Code: {response.status_code}")

        if response.status_code == 200:
            print("Anomaly detection successful!")
            response_json = response.json()
            print("Response JSON:", json.dumps(response_json, indent=2))
            if response_json.get("anomalies"):
                print("\n✅ Test Passed: Anomaly was correctly identified.")
            else:
                print("\n❌ Test Failed: Anomaly was NOT identified.")
        else:
            print("Anomaly detection failed.")
            print("Error Response:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while connecting to the API server: {e}")
        print("Please ensure the API server is running correctly.")

if __name__ == "__main__":
    # !!! 替换为您的API密钥 !!!
    # 您可以通过访问 http://127.0.0.1:8000/docs#/Admin%20Operations/create_api_key_admin_api_keys_post 生成一个新密钥
    YOUR_API_KEY = "3369df94-7513-459e-be83-104bdb046b85" 

    if not YOUR_API_KEY:
        print("Error: YOUR_API_KEY is not set. Please set your API key in api_client_test.py")
    else:
        # 运行预测测试
        run_prediction_test(YOUR_API_KEY)

        # 运行异常检测测试
        run_anomaly_detection_test(YOUR_API_KEY)