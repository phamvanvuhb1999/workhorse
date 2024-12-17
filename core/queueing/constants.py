from enum import Enum


class WorkerStatuses:
    HEALTHY: str = "healthy"
    UNHEALTHY: str = "unhealthy"


class QueueNames:
    WAITING: str = "waiting"
    CELERY: str = "celery"


class TaskNames(Enum):
    MODEL_INVOKE: str = "model_invoke"
    PUSH_TO_RESOURCE_MANAGER: str = "push_to_resource_manager"
    MODEL_RESPONSE_TIME_ORDER: str = "model_response_time_order"


class TaskRetryConfigs:
    INFINITY_RETRY_TIMES = None
    DEFAULT_RETRY_TIMES = 2
    STANDARD_RETRY_TIMES = 5
    HIGH_RETRY_TIMES = 15

    DEFAULT_RETRY_COUNTDOWN = 180
    STANDARD_RETRY_COUNTDOWN = 30
    SHORT_RETRY_COUNTDOWN = 10
    SUPER_SHORT_RETRY_COUNTDOWN = 0.15


class TrackingCategories:
    RETRIES: str = "retries"
    SUCCESS: str = "success"
    FAILURE: str = "failure"

    TIME_WINDOW_TASKS: str = "sent-tasks-current-time-window"


ROUTING_CACHE_SECONDS = 60
WORKERS_CACHE_SECONDS = 3600
