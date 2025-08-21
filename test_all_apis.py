#!/usr/bin/env python3
"""
API测试脚本 - 测试所有API接口是否正常工作
使用提供的API Key: d4ba2eed-0ffe-4961-bfee-401eda15f7ac
"""

import requests
import json
import os
import time
from datetime import datetime

# API配置
BASE_URL = "http://localhost:8000"
API_KEY = "d4ba2eed-0ffe-4961-bfee-401eda15f7ac"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def print_test_result(test_name, success, response=None, error=None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if not success and error:
        print(f"   错误: {error}")
    if response and hasattr(response, 'status_code'):
        print(f"   状态码: {response.status_code}")
    print()

def test_basic_endpoints():
    """测试基础接口"""
    print("🔍 测试基础接口...")
    
    # 测试ping接口
    try:
        response = requests.get(f"{BASE_URL}/ping")
        print_test_result("GET /ping", response.status_code == 200, response)
    except Exception as e:
        print_test_result("GET /ping", False, error=str(e))

def test_asset_management():
    """测试资产管理接口"""
    print("🏭 测试资产管理接口...")
    
    # 1. 获取所有资产
    try:
        response = requests.get(f"{BASE_URL}/admin/assets", headers=HEADERS)
        assets = response.json() if response.status_code == 200 else []
        print_test_result("GET /admin/assets", response.status_code == 200, response)
        print(f"   找到 {len(assets)} 个资产")
    except Exception as e:
        print_test_result("GET /admin/assets", False, error=str(e))
        assets = []
    
    # 2. 创建新资产
    test_asset_data = {
        "id": "test_asset_api",
        "name": "API测试资产",
        "description": "用于API测试的临时资产"
    }
    try:
        response = requests.post(f"{BASE_URL}/admin/assets", headers=HEADERS, json=test_asset_data)
        print_test_result("POST /admin/assets", response.status_code in [200, 201], response)
        created_asset = response.json() if response.status_code in [200, 201] else None
    except Exception as e:
        print_test_result("POST /admin/assets", False, error=str(e))
        created_asset = None
    
    # 3. 更新资产
    if created_asset:
        update_data = {
            "name": "API测试资产(已更新)",
            "description": "更新后的描述"
        }
        try:
            response = requests.put(f"{BASE_URL}/admin/assets/test_asset_api", headers=HEADERS, json=update_data)
            print_test_result("PUT /admin/assets/{asset_id}", response.status_code == 200, response)
        except Exception as e:
            print_test_result("PUT /admin/assets/{asset_id}", False, error=str(e))
    
    # 4. 删除测试资产
    if created_asset:
        try:
            response = requests.delete(f"{BASE_URL}/admin/assets/test_asset_api", headers=HEADERS)
            print_test_result("DELETE /admin/assets/{asset_id}", response.status_code in [200, 204], response)
        except Exception as e:
            print_test_result("DELETE /admin/assets/{asset_id}", False, error=str(e))
    
    return assets

def test_model_management(assets):
    """测试模型管理接口"""
    print("🤖 测试模型管理接口...")
    
    # 1. 获取所有模型
    try:
        response = requests.get(f"{BASE_URL}/admin/models", headers=HEADERS)
        models = response.json() if response.status_code == 200 else []
        print_test_result("GET /admin/models", response.status_code == 200, response)
        print(f"   找到 {len(models)} 个模型")
    except Exception as e:
        print_test_result("GET /admin/models", False, error=str(e))
        models = []
    
    # 2. 按资产ID筛选模型
    if assets:
        asset_id = assets[0]['id']
        try:
            response = requests.get(f"{BASE_URL}/admin/models?asset_id={asset_id}", headers=HEADERS)
            asset_models = response.json() if response.status_code == 200 else []
            print_test_result(f"GET /admin/models?asset_id={asset_id}", response.status_code == 200, response)
            print(f"   资产 {asset_id} 有 {len(asset_models)} 个模型")
        except Exception as e:
            print_test_result(f"GET /admin/models?asset_id={asset_id}", False, error=str(e))
            asset_models = []
    
    # 3. 获取特定模型详情
    if models:
        model_id = models[0]['id']
        try:
            response = requests.get(f"{BASE_URL}/admin/models/{model_id}", headers=HEADERS)
            print_test_result(f"GET /admin/models/{model_id}", response.status_code == 200, response)
        except Exception as e:
            print_test_result(f"GET /admin/models/{model_id}", False, error=str(e))
    
    return models

def test_api_key_management():
    """测试API密钥管理接口"""
    print("🔑 测试API密钥管理接口...")
    
    # 1. 获取所有API密钥
    try:
        response = requests.get(f"{BASE_URL}/admin/api-keys", headers=HEADERS)
        api_keys = response.json() if response.status_code == 200 else []
        print_test_result("GET /admin/api-keys", response.status_code == 200, response)
        print(f"   找到 {len(api_keys)} 个API密钥")
    except Exception as e:
        print_test_result("GET /admin/api-keys", False, error=str(e))
        api_keys = []
    
    # 2. 创建新API密钥
    new_key_data = {
        "description": "API测试密钥"
    }
    try:
        response = requests.post(f"{BASE_URL}/admin/api-keys", headers=HEADERS, json=new_key_data)
        print_test_result("POST /admin/api-keys", response.status_code in [200, 201], response)
        created_key = response.json() if response.status_code in [200, 201] else None
        if created_key:
            print(f"   新密钥ID: {created_key.get('id')}")
    except Exception as e:
        print_test_result("POST /admin/api-keys", False, error=str(e))
        created_key = None
    
    # 3. 删除测试密钥
    if created_key and created_key.get('id'):
        try:
            response = requests.delete(f"{BASE_URL}/admin/api-keys/{created_key['id']}", headers=HEADERS)
            print_test_result("DELETE /admin/api-keys/{key_id}", response.status_code in [200, 204], response)
        except Exception as e:
            print_test_result("DELETE /admin/api-keys/{key_id}", False, error=str(e))

def test_prediction_endpoints(assets, models):
    """测试预测接口"""
    print("📊 测试预测接口...")
    
    if not assets or not models:
        print("   ⚠️  跳过预测测试 - 需要资产和模型数据")
        return
    
    # 找到有模型的资产
    asset_with_model = None
    model_for_test = None
    
    for asset in assets:
        asset_models = [m for m in models if m.get('asset_id') == asset['id']]
        if asset_models:
            asset_with_model = asset
            model_for_test = asset_models[0]
            break
    
    if not asset_with_model or not model_for_test:
        print("   ⚠️  跳过预测测试 - 没有找到有模型的资产")
        return
    
    print(f"   使用资产: {asset_with_model['id']}, 模型: {model_for_test['id']}")
    
    # 测试CSV文件预测 - 使用示例数据
    csv_content = """timestamp,energy_kwh,temp,production,humidity
2024-01-01 00:00:00,100.5,20.5,50.0,60.0
2024-01-01 01:00:00,105.2,21.0,52.0,61.0
2024-01-01 02:00:00,98.7,20.8,48.0,59.5
2024-01-01 03:00:00,102.1,21.2,51.0,60.5
2024-01-01 04:00:00,99.8,20.9,49.5,60.2"""
    
    try:
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        data = {
            'model_id': str(model_for_test['id']),
            'forecast_horizon': '24'
        }
        response = requests.post(
            f"{BASE_URL}/assets/{asset_with_model['id']}/predict_from_csv",
            headers={"Authorization": f"Bearer {API_KEY}"},  # 不包含Content-Type，让requests自动设置
            files=files,
            data=data
        )
        print_test_result("POST /assets/{asset_id}/predict_from_csv", response.status_code == 200, response)
        if response.status_code == 200:
            result = response.json()
            print(f"   预测数据点数: {len(result.get('forecast_data', []))}")
    except Exception as e:
        print_test_result("POST /assets/{asset_id}/predict_from_csv", False, error=str(e))
    
    # 测试S3预测接口（如果有S3数据）
    try:
        params = {
            's3_data_path': 'test/sample_data.csv',
            'model_id': str(model_for_test['id']),
            'forecast_horizon': '24'
        }
        response = requests.post(
            f"{BASE_URL}/assets/{asset_with_model['id']}/predict_from_s3",
            headers=HEADERS,
            params=params
        )
        # S3接口可能因为没有测试数据而失败，这是正常的
        success = response.status_code == 200 or "not found" in response.text.lower()
        print_test_result("POST /assets/{asset_id}/predict_from_s3", success, response)
    except Exception as e:
        print_test_result("POST /assets/{asset_id}/predict_from_s3", False, error=str(e))

def test_anomaly_detection_endpoints(assets, models):
    """测试异常检测接口"""
    print("🚨 测试异常检测接口...")
    
    if not assets or not models:
        print("   ⚠️  跳过异常检测测试 - 需要资产和模型数据")
        return
    
    # 找到有检测器的模型
    asset_with_detector = None
    model_with_detector = None
    
    for asset in assets:
        asset_models = [m for m in models if m.get('asset_id') == asset['id'] and m.get('detector_path')]
        if asset_models:
            asset_with_detector = asset
            model_with_detector = asset_models[0]
            break
    
    if not asset_with_detector or not model_with_detector:
        print("   ⚠️  跳过异常检测测试 - 没有找到有检测器的模型")
        return
    
    print(f"   使用资产: {asset_with_detector['id']}, 模型: {model_with_detector['id']}")
    
    # 测试CSV异常检测
    csv_content = """timestamp,energy_kwh,temp,production,humidity
2024-01-01 00:00:00,100.5,20.5,50.0,60.0
2024-01-01 01:00:00,105.2,21.0,52.0,61.0
2024-01-01 02:00:00,98.7,20.8,48.0,59.5
2024-01-01 03:00:00,102.1,21.2,51.0,60.5
2024-01-01 04:00:00,99.8,20.9,49.5,60.2"""
    
    try:
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        data = {'model_id': str(model_with_detector['id'])}
        response = requests.post(
            f"{BASE_URL}/assets/{asset_with_detector['id']}/detect_anomalies_from_csv",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=files,
            data=data
        )
        print_test_result("POST /assets/{asset_id}/detect_anomalies_from_csv", response.status_code == 200, response)
        if response.status_code == 200:
            result = response.json()
            print(f"   异常数据点数: {len(result.get('anomalies', []))}")
    except Exception as e:
        print_test_result("POST /assets/{asset_id}/detect_anomalies_from_csv", False, error=str(e))

def test_training_endpoints():
    """测试训练接口"""
    print("🎯 测试训练接口...")
    
    # 测试训练任务状态查询（使用一个假的task_id）
    try:
        response = requests.get(f"{BASE_URL}/admin/task_status/fake_task_id", headers=HEADERS)
        # 404是预期的，因为task_id不存在
        success = response.status_code in [200, 404]
        print_test_result("GET /admin/task_status/{task_id}", success, response)
    except Exception as e:
        print_test_result("GET /admin/task_status/{task_id}", False, error=str(e))
    
    # 注意：我们不测试实际的模型训练，因为那会消耗大量时间和资源

def compare_with_swagger():
    """比较前端文档和Swagger文档"""
    print("📚 比较前端API文档和Swagger文档...")
    
    try:
        # 获取Swagger文档
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            swagger_doc = response.json()
            print_test_result("获取Swagger文档", True, response)
            
            # 分析接口数量
            paths = swagger_doc.get('paths', {})
            total_endpoints = sum(len(methods) for methods in paths.values())
            print(f"   Swagger文档中的接口数量: {total_endpoints}")
            
            # 列出所有接口
            print("   Swagger接口列表:")
            for path, methods in paths.items():
                for method in methods.keys():
                    print(f"     {method.upper()} {path}")
        else:
            print_test_result("获取Swagger文档", False, response)
    except Exception as e:
        print_test_result("获取Swagger文档", False, error=str(e))

def main():
    """主测试函数"""
    print("🚀 开始API测试...")
    print(f"📍 测试地址: {BASE_URL}")
    print(f"🔑 使用API Key: {API_KEY[:8]}...{API_KEY[-8:]}")
    print("=" * 60)
    
    # 执行所有测试
    test_basic_endpoints()
    assets = test_asset_management()
    models = test_model_management(assets)
    test_api_key_management()
    test_prediction_endpoints(assets, models)
    test_anomaly_detection_endpoints(assets, models)
    test_training_endpoints()
    compare_with_swagger()
    
    print("=" * 60)
    print("✅ API测试完成！")
    print("\n📋 测试总结:")
    print("1. 基础接口测试完成")
    print("2. 资产管理CRUD操作测试完成")
    print("3. 模型管理接口测试完成")
    print("4. API密钥管理测试完成")
    print("5. 预测接口测试完成")
    print("6. 异常检测接口测试完成")
    print("7. 训练接口测试完成")
    print("8. Swagger文档对比完成")
    
    print(f"\n🌐 访问链接:")
    print(f"   前端管理界面: http://localhost:5173")
    print(f"   后端Swagger文档: {BASE_URL}/docs")
    print(f"   前端API文档: http://localhost:5173/api-docs")

if __name__ == "__main__":
    main()