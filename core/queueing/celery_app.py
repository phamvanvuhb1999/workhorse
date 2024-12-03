from datetime import datetime
from functools import partial

from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from configs.common import settings
from core.queueing.base import BaseTask
from core.queueing.constants import QueueNames

celery_app = Celery("telehealth-ai-ocr-tasks-queues")
celery_app.task_cls = BaseTask
celery_app.config_from_object("core.queueing.celery_config")
celery_app.conf.timezone = "Asia/Singapore"
celery_app.conf.task_default_queue = QueueNames.CELERY
celery_app.conf.task_default_exchange_type = "direct"
celery_app.conf.task_default_routing_key = QueueNames.CELERY


# Task queues
celery_app.conf.task_queues = (
    # Add queues here
    Queue(QueueNames.CELERY, routing_key=QueueNames.CELERY),
    Queue(QueueNames.WAITING, routing_key=QueueNames.WAITING),
)

# Routing task with simple queue name rule
celery_app.conf.task_routes = {
    # Add routing here
}

celery_app.conf.task_default_rate_limit = '100000/s'
celery_app.conf.task_always_eager = settings.TASK_ALWAYS_EAGER
celery_app.conf.task_eager_propagates = settings.TASK_EAGER_PROPAGATES

celery_app.conf.broker_transport_options = {
    'visibility_timeout': settings.BROKER_VISIBLE_TIMEOUT,  # 1 hour timeout for task visibility
    'socket_keepalive_options': {
        'keepalive': True,
        'keepalive_interval': settings.SOCKET_KEEPALIVE_INTERVAL,  # Send keep-alive messages every 60 seconds
        'keepalive_count': settings.SOCKET_KEEPALIVE_COUNT,  # Number of keep-alive probes
    },
}

# Task Priority
# celery_app.conf.broker_transport_options = {
#     'priority_steps': list(range(10)),
#     'sep':':',
#     'queue_order_strategy':'priority',
# }

def nowfunc():  # noqa
    return datetime.now()


cet_crontab = partial(crontab, nowfun=nowfunc)

celery_app.conf.beat_schedule = {
    # Add cronjob here
}

celery_app.autodiscover_tasks([
    "core.queueing.tasks",
])
