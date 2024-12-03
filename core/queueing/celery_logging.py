import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from celery.utils.log import get_task_logger

from configs.common import settings


logger = get_task_logger(__name__)


os.makedirs(settings.LOG_DIR, exist_ok=True)

log_handler = TimedRotatingFileHandler(
    os.path.join(settings.LOG_DIR, f'celery_sync_{datetime.now().strftime("%Y-%m-%d")}.log'),
    when="midnight",
    interval=1,
    backupCount=7,
)

log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    ),
)

logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)
