# 工业能耗预测与异常检测全栈应用

本项目是一个功能完整的全栈应用，旨在提供工业级的能源消耗预测和实时异常检测服务。它整合了先进的机器学习模型、一个健壮的后端 API、一个异步任务处理系统以及一个用户友好的前端管理界面。

## ✨ 核心功能

- **多种预测模型**: 支持多种业界领先的时间序列预测模型，包括 `LightGBM`, `TiDE`, `LSTM`, 和 `TFT`。
- **实时异常检测**: 基于预测残差和 `QuantileDetector`，能够实时识别数据流中的异常能耗点。
- **异步模型训练**: 利用 `Celery` 和 `Redis`，在后台异步处理耗时的模型训练任务，避免阻塞 API 服务。
- **RESTful API**: 提供了一套完整的 `FastAPI` 接口，用于模型管理、触发训练、执行预测和异常检测。
- **现代化管理前端**: 基于 `React` 和 `Material-UI` 构建，提供模型管理、数据上传、任务触发以及结果可视化的完整操作界面。
- **容器化部署**: 通过 `Docker` 和 `Docker Compose`，实现整个系统（包括数据库、缓存、对象存储和应用服务）的一键化部署和管理。
- **对象存储集成**: 使用 `MinIO` (S3 兼容) 作为模型文件、数据集和其他产物的存储后端，实现生产级的文件管理。

## 🏗️ 技术架构

系统采用解耦的微服务架构，各个组件各司其职，并通过网络进行通信。

### 技术选型

| 分类         | 技术                               |
| :----------- | :--------------------------------- |
| **前端**     | React, Material-UI, Recharts, Vite |
| **后端**     | Python, FastAPI                    |
| **ML 框架**  | Darts (u8darts), PyTorch, Scikit-learn |
| **数据库**   | PostgreSQL                         |
| **数据库迁移** | Alembic                            |
| **异步任务** | Celery, Redis                      |
| **对象存储** | MinIO (S3 兼容)                    |
| **部署**     | Docker, Docker Compose             |

### 架构图

```mermaid
graph TD
    subgraph "用户端"
        User[👨‍💻 用户/管理员]
    end

    subgraph "应用服务 (Docker Compose)"
        Frontend[🌐 React 前端<br>(Nginx/Vite)]
        API[🚀 FastAPI 后端 API]
        Worker[👷 Celery Worker]
    end

    subgraph "基础设施 (Docker Compose)"
        DB[(🐘 PostgreSQL)]
        Cache[(⚡ Redis)]
        S3[📦 MinIO/S3]
    end

    User -- "访问/操作" --> Frontend
    Frontend -- "HTTP API 请求" --> API

    API -- "读写模型元数据" --> DB
    API -- "发送训练任务" --> Cache
    API -- "加载模型/数据" --> S3
    API -- "返回结果" --> Frontend

    Worker -- "获取训练任务" --> Cache
    Worker -- "读写模型元数据" --> DB
    Worker -- "下载数据/上传模型" --> S3
```

## 🚀 本地运行指南 (Getting Started)

通过 Docker Compose，您可以轻松地在本地一键启动整个应用。

### 1. 环境准备

- 确保您的机器上已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)。

### 2. 环境配置

- 项目根目录下有一个 `.env.example` 文件。请复制它来创建一个 `.env` 文件：
  ```bash
  cp .env.example .env
  ```
- 您可以根据需要修改 `.env` 文件中的配置，但默认值已足够用于本地开发。

### 3. 构建并启动服务

- 在项目根目录下，执行以下命令来构建镜像并启动所有服务：
  ```bash
  docker-compose up --build
  ```
- 该命令会启动 API 服务、Celery Worker、前端、数据库、Redis 和 MinIO。数据库初始化任务 (`db-init`) 会自动运行，并使用 Alembic 将数据库结构更新到最新版本。

### 4. 创建初始 API 密钥

- 为了能与受保护的 API 端点交互，您需要创建一个初始的 API 密钥。
- 待服务启动后，打开一个新的终端，执行以下命令：
  ```bash
  docker-compose run --rm api python -m tools.create_initial_key "My First Key"
  ```
- **请务必复制并保存好输出的 API 密钥**，前端页面的所有请求都需要使用它。

### 5. 访问系统

- **前端管理界面**:
  - 访问 `http://localhost:5173`
  - 在页面的 API 密钥输入框中，粘贴上一步生成的密钥。
- **后端 API 文档 (Swagger UI)**:
  - 访问 `http://localhost:8000/docs`
  - 您可以在这里浏览所有 API 端点，并进行交互式测试。
- **MinIO 对象存储控制台**:
  - 访问 `http://localhost:9001`
  - 使用 `.env` 文件中定义的 `MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD` 登录。默认情况下，`models` bucket 会被自动创建。

## 🛠️ 使用说明

1.  **访问前端**: 打开 `http://localhost:5173` 并输入您的 API 密钥。
2.  **创建资产**: 在 "模型训练" 标签页，首先创建一个资产（如 `production_line_A`）。
3.  **上传数据**: 为该资产上传用于训练的 CSV 数据文件。
4.  **触发训练**: 选择模型类型并点击 "开始训练"。您可以在终端查看 `worker` 服务的日志来跟踪进度。
5.  **执行预测/异常检测**: 训练完成后，切换到 "能源预测" 或 "异常检测" 标签页，选择刚刚训练好的模型，上传一份用于推理的 CSV 文件，即可看到可视化的结果。

## 📂 项目结构

```
/
├── admin_frontend/ # React 前端应用
├── alembic/        # Alembic 数据库迁移脚本
├── api_server/     # FastAPI 后端服务
├── celery_worker/  # Celery 异步任务定义
├── core/           # 核心的训练和推理服务逻辑
├── database/       # SQLAlchemy 模型和数据库配置
├── tools/          # 实用工具脚本 (如创建API密钥)
├── .env.example    # 环境变量示例文件
├── docker-compose.yml # Docker Compose 配置文件
├── Dockerfile      # 应用的 Dockerfile
└── README.md
```

## 📄 许可证

本项目根据 [MIT License](LICENSE) 的条款进行许可。
