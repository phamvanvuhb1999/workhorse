from configs.common import settings


# Celery configs
broker_url = settings.CELERY_BROKER_URL

result_backend = settings.CELERY_RESULT_BACKEND or settings.REDIS_URL

# ACK configs
task_reject_on_worker_lost = settings.TASK_REJECT_ON_WORKER_LOST

acks_late = settings.TASK_ACKS_LATE

# Time limit configs
task_soft_time_limit = settings.TASK_SOFT_TIME_LIMIT

task_time_limit = settings.TASK_TIME_LIMIT

result_expires = settings.CELERY_RESULT_EXPIRES

broker_connection_retry_on_startup = settings.BROKER_CONNECTION_RETRY_ON_STARTUP

worker_concurrency = settings.WORKER_CONCURRENCY

# Prefetch tasks (prefetch_nums = worker_concurrency * worker_prefetch_multiplier)
# Each worker will prefetch one task before
worker_prefetch_multiplier = settings.WORKER_PREFETCH_MULTIPLIER

worker_max_tasks_per_child = settings.WORKER_MAX_TASKS_PER_CHILD

accept_content = ["application/json"]

result_serializer = "json"

task_serializer = "json"

enable_utc = settings.WORKER_ENABLE_UTC

# Task annotations:
task_annotations = {
    # Add annotation for tasks here (ratelimit,...)
}
