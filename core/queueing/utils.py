import contextlib
import logging
from datetime import datetime

from configs.common import settings
from core.cache.async_redis_client import AsyncRedisClient
from core.cache.key_generator import QueueTrackingKeyGen
from core.cache.sync_redis_client import SyncRedisClient
from core.exceptions.queueing import MessageOrderingException
from core.queueing.constants import TrackingCategories
from core.utils.convert_helper import save_convert


logger = logging.getLogger(__name__)


class TaskTracking:
    def __init__(self, time_window: int, **kwargs):
        self.time_window = time_window
        self.cache_client = SyncRedisClient.get_instance(**kwargs)

    @classmethod
    def get_score(cls, is_str: bool = True):
        score = datetime.now().timestamp()
        if is_str:
            score = str(score)
        return score

    def get_max_score(self):
        return str(self.get_score(is_str=False) - self.time_window)

    def log(self, tracking_key: str, task_id: str):
        score = self.get_score()
        self.cache_client.zadd(tracking_key, task_id, score)

    def count(self, tracking_key: str):
        max_score: str = self.get_max_score()
        return self.cache_client.zcard(tracking_key, with_validate=True, max_score=max_score)


class OrderingTracking:
    def __init__(self, time_window: int = 10, **kwargs):
        self.time_window = time_window
        self.cache_client = SyncRedisClient.get_instance(**kwargs)

    def incr(self, tracking_key: str, model_id: str, amount: float = 1, **kwargs):
        return self.cache_client.zincrby(key=tracking_key, amount=amount, value=model_id, **kwargs)

    def decr(self, tracking_key: str, model_id: str, amount: float = 1, **kwargs):
        return self.cache_client.zincrby(key=tracking_key, amount=-amount, value=model_id, **kwargs)

    def range_by_score(self, tracking_key: int, batch_num: int = 10):
        start_index = 0

        while True:
            end_index = start_index + batch_num
            result = self.cache_client.zrange(key=tracking_key, start=start_index, end=end_index)
            with contextlib.suppress(Exception):
                yield result
            if not result:
                break
            start_index = end_index + 1


class TaskRateLimit(TaskTracking):
    def queue_limit_exceeded(self, queue_name: str):
        exceeded: bool = True

        try:
            # Get task executed number
            success_speed = self.count(
                QueueTrackingKeyGen.gen_collection_key(
                    queue=queue_name,
                    category=TrackingCategories.SUCCESS,
                ),
            )
            retry_speed = self.count(
                QueueTrackingKeyGen.gen_collection_key(
                    queue=queue_name,
                    category=TrackingCategories.RETRIES,
                ),
            )
            # Get task enqueue number
            enqueue_speed = self.count(
                QueueTrackingKeyGen.gen_collection_key(
                    queue=queue_name,
                    category=TrackingCategories.TIME_WINDOW_TASKS,
                ),
            )
            # Get number of tasks in queue
            pending_task: int = self.cache_client.llen(queue_name)

            # Calculate statistic in default 10s
            available_pos = (
                settings.MAX_TASK_PER_QUEUES
                + success_speed
                + retry_speed
                - pending_task
                - enqueue_speed
            )

            if available_pos > 0:
                exceeded = False
        except Exception as e:
            logger.error(f"Can not get worker limit information, due to {e}")
        return exceeded

    def update_task_limit(self, queue: str, category: str, task_id: str, **kwargs):
        self.log(
            QueueTrackingKeyGen.gen_collection_key(
                queue=queue,
                category=category,
            ),
            task_id=task_id,
        )


class MessageOrderingService:
    @classmethod
    async def tracking(cls, tracking_key: str, item: str = None, score: str = None):
        if score is not None:
            cache_client = AsyncRedisClient.get_instance()

            await cache_client.async_zadd(
                tracking_key,
                item=item or score,
                score=score,
            )
            await cache_client.async_expire(tracking_key, settings.TASK_ROUTING_FOOTPRINT_TIME)

    @classmethod
    def verify_message_order(cls, tracking_key: str, score: str, remove_old: bool = False):
        if tracking_key and score:
            cache_client = SyncRedisClient.get_instance()
            if (
                cache_client.zcount(tracking_key, max_score=f"({score}") > 0
                and datetime.now().timestamp() - save_convert(score, float) < settings.MESSAGE_RESPONSE_ALLOW_DELAY_TIME
            ):
                raise MessageOrderingException()
            else:
                if remove_old:
                    cls.remove_old_item(tracking_key, score)

    @classmethod
    def remove_old_item(cls, tracking_key: str, score: str):
        cache_client = SyncRedisClient.get_instance()

        cache_client.zremrangebyscore(tracking_key, max_score=score)
