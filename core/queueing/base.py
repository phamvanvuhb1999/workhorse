import asyncio
import logging
import os
import signal
from typing import Any

from celery import Task

from configs.common import settings
from core.queueing.constants import TrackingCategories
from core.queueing.utils import TaskRateLimit
from core.queueing.utils import TaskTracking


logger = logging.getLogger(__name__)

task_tracker = TaskTracking(time_window=settings.STATISTIC_TRACKING_TIME_WINDOW)

circuit_breaker_limiter = TaskRateLimit(time_window=settings.QUEUES_TRACKING_TIME_WINDOW)


class BaseTask(Task):
    def handle_soft_time_limit_exceeded(self, *args, **kwargs):
        # Log the soft time limit exceeded event
        logger.error("Soft time limit exceeded for task %s", self.request.id)

        max_retries_time: int = kwargs.get("max_retries_time", self.max_retries)
        kwargs.update(max_retries_time=max(max_retries_time - 1, 0))

        # Push the task back to the queue
        self.apply_async(args=args, kwargs=kwargs, routing_key=self.request.delivery_info.get('routing_key'))

        # Get the current worker PID
        pid = os.getpid()
        # Terminate the current process
        os.kill(pid, signal.SIGTERM)

    def on_success(self, retval, task_id, args, kwargs):
        routing_key = self.request.delivery_info.get("routing_key")

        circuit_breaker_limiter.update_task_limit(
            queue=routing_key,
            category=TrackingCategories.SUCCESS,
            task_id=task_id,
            **kwargs,
        )

    def retry(self, *args, **kwargs):
        task = super().retry(*args, **kwargs)

        routing_key = self.request.delivery_info.get("routing_key")

        circuit_breaker_limiter.update_task_limit(
            queue=routing_key,
            category=TrackingCategories.RETRIES,
            task_id=task.id,
            **kwargs,
        )


    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if self.request.retries == self.max_retries:
            routing_key = self.request.delivery_info.get("routing_key")

            circuit_breaker_limiter.update_task_limit(
                queue=routing_key,
                category=TrackingCategories.FAILURE,
                task_id=task_id,
                **kwargs,
            )

    def apply_async(
        self,
        args: Any = None,
        kwargs: Any = None,
        **option,
    ):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []

        rate_limiter = TaskRateLimit(time_window=settings.QUEUES_TRACKING_TIME_WINDOW)

        rt_id_only: bool = option.get("rt_id_only", True)

        routing_key = kwargs.get("routing_key")

        if routing_key and rate_limiter.queue_limit_exceeded(routing_key):
            return
        else:
            job = super().apply_async(args=args, kwargs=kwargs, **option)
            logger.info(f"Task {self.name} sent with ID: {job.id}")

            # Update rate_limiter
            rate_limiter.update_task_limit(
                category=TrackingCategories.TIME_WINDOW_TASKS,
                queue=routing_key,
                task_id=job.id,
            )
            if rt_id_only:
                return job.id
            else:
                return job
