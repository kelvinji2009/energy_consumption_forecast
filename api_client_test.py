import requests
import json
from datetime import datetime, timedelta

def run_prediction_test():
    """向本地运行的API服务发送一个测试预测请求。"""

    print("--- API Client Test ---")

    # API的URL
    asset_id = "production_line_A"
    url = f"http://127.0.0.1:8000/assets/{asset_id}/predict"

    # 1. 构造一些模拟的历史数据
    # 关键修复：模型需要至少`input_chunk_length` (168) 个历史数据点来构建特征。
    # 我们现在生成168个点。
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

    # 2. 构造请求体
    request_payload = {
        "historical_data": historical_data,
        "forecast_horizon": 12 # 要求预测未来12小时
    }

    print(f"Sending POST request to: {url}")
    print(f"Historical data points sent: {len(historical_data)}")

    # 3. 发送POST请求
    try:
        response = requests.post(url, json=request_payload, timeout=10)

        # 4. 打印响应结果
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

if __name__ == "__main__":
    run_prediction_test()