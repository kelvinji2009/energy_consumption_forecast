from celery import Celery
import os

# Redis broker URL. In production, this should be an environment variable.
# For local development, we assume Redis is running on default port.
REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "energy_forecast_tasks",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
    include=["celery_worker.tasks"] # This tells Celery where to find tasks
)

# Optional: Celery configuration
celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai', # Set timezone for tasks
    enable_utc=True,
)

if __name__ == "__main__":
    # This block is for running the worker directly for testing purposes
    # In a real deployment, you'd use `celery -A celery_worker.celery_app worker -l info`
    print("Starting Celery worker (for development/testing)...")
    celery_app.worker_main(['worker', '--loglevel=info'])
