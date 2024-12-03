from typing import Optional

from pathlib import Path
from dotenv import load_dotenv
from pydantic import RedisDsn
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from core.enumerates.common import Environment


load_dotenv()


BASE_DIR = str(Path(__file__).parent.parent.absolute())


class Config(BaseSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.REDIS_URL:
            self.REDIS_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


    PORT: int = 8080

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_TTL: int = 300
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_URL: Optional[RedisDsn] | str | None = None

    # Celery
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    # Web Application
    CORS_ORIGINS: Optional[str] = ""
    CORS_ORIGINS_REGEX: Optional[str] = None
    CORS_HEADERS: Optional[str] = ""
    ENV_NAME: Optional[str] = Environment.STAGING
    SITE_DOMAIN: str = "localhost"

    # JWT
    JWT_ALG: str = "HS256"
    JWT_SECRET: str = ""
    JWT_EXPIRE_SECONDS: int = 900  # By default, after 900 seconds, token will be expired
    JWT_REFRESH_TOKEN_EXPIRE_SECONDS: int = 3600

    WEBSOCKET_SESSION_EXPIRE_SECONDS: int = 3600  # session be expired in 1 hour if not active
    RECOMMEND_KEY_EXPIRE_TIME_SECONDS: int = 300

    SECURE_COOKIES: bool = False

    TESTING: Optional[bool] = False
    DEBUG: Optional[bool] = False

    # Queue settings
    MAX_TASK_PER_QUEUES: int = 1000
    MAX_TASK_PER_USER: int = 20  # user can send 20 tasks each 10 seconds
    QUEUES_TRACKING_TIME_WINDOW: int = 10  # each 10 seconds
    STATISTIC_TRACKING_TIME_WINDOW: int = 86400  # 1 day

    # Worker and task config envs
    WORKER_CONCURRENCY: int | None = None
    WORKER_PREFETCH_MULTIPLIER: int = 4
    TASK_SOFT_TIME_LIMIT: int = 60
    TASK_TIME_LIMIT: int = 1200
    WORKER_MAX_TASKS_PER_CHILD: int = 1000
    TASK_ALWAYS_EAGER: bool = False
    TASK_EAGER_PROPAGATES: bool = False
    TASK_REJECT_ON_WORKER_LOST: bool = True
    TASK_ACKS_LATE: bool = True
    CELERY_RESULT_EXPIRES: int = 3600

    MODEL_INFERENCE_TIMEOUT: int = 60

    BROKER_CONNECTION_RETRY_ON_STARTUP: bool = True
    WORKER_ENABLE_UTC: bool = True

    # Broker configs
    BROKER_VISIBLE_TIMEOUT: int = 3600
    SOCKET_KEEPALIVE_INTERVAL: int = 60
    SOCKET_KEEPALIVE_COUNT: int = 5

    TASK_ROUTING_FOOTPRINT_TIME: int = 3600

    # Message synchronize config
    MESSAGE_RESPONSE_ALLOW_DELAY_TIME: int = 60

    log_dir: str = "logs"
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Config()
