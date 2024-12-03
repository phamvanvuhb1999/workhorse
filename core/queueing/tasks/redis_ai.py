from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded

from core.cache.sync_redis_client import SyncRedisClient
from core.ai_resource_manager import RedisAIModel
from core.queueing import celery_app
from core.queueing.constants import TaskNames, TaskRetryConfigs
from core.ai_resource_manager.resource_manager import AIResourceManager
from core.utils import T5InferenceHelper


@celery_app.task(
    bind=True,
    name=TaskNames.PUSH_TO_RESOURCE_MANAGER.value,
    autoretry_for=(Exception,),
    max_retries=TaskRetryConfigs.INFINITY_RETRY_TIMES,
    default_retry_delay=TaskRetryConfigs.SUPER_SHORT_RETRY_COUNTDOWN,
)
def push_to_source_manager(
    self,
    *args,
    **kwargs,
):
    AIResourceManager.get_instance().run(task_id=self.request.id, *args, **kwargs)


@celery_app.task(
    name=TaskNames.MODEL_INVOKE.value,
    autoretry_for=(SoftTimeLimitExceeded, TimeLimitExceeded),
    max_retries=TaskRetryConfigs.DEFAULT_RETRY_TIMES,
    default_retry_delay=TaskRetryConfigs.STANDARD_RETRY_COUNTDOWN,
)
def model_invoke(
    model_type: str,
    model_index: str,
    indices: Any,
    **kwargs,
):
    reply: str = ""
    model_class = RedisAIModel.get_submodels_map().get(model_type)
    if model_class is not None:
        model = model_class.get_instance(prefix=f"{model_class.class_prefix()}:{model_index}")
        numpy_array = T5InferenceHelper.list2numpy(indices)
        reply = model.process(numpy_array)
        model.client.publish("all", message=reply)

    block_key = kwargs.get("block_key")
    block_identify = kwargs.get("block_identify")
    if block_key and block_identify:
        redis_client = SyncRedisClient.get_instance()

        if redis_client.get(block_key) == block_identify:
            redis_client.delete(key=block_identify)
    return reply
