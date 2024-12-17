import pickle
from datetime import datetime

from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded

from core.ai_resource_manager import RedisAIModel
from core.cache.key_generator import RedisAIModelKeyGen
from core.cache.sync_redis_client import SyncRedisClient
from core.queueing import celery_app
from core.queueing.constants import TaskNames, TaskRetryConfigs
from core.ai_resource_manager.resource_manager import AIResourceManager
from core.queueing.utils import OrderingTracking


@celery_app.task(
    name=TaskNames.MODEL_RESPONSE_TIME_ORDER.value,
    autoretry_for=(SoftTimeLimitExceeded, TimeLimitExceeded),
    max_retries=TaskRetryConfigs.INFINITY_RETRY_TIMES,
    default_retry_delay=TaskRetryConfigs.SUPER_SHORT_RETRY_COUNTDOWN,
)
def model_response_time_order(
    tracking_key: str = None,
    model_index: str = None,
    score: float = None,
    **kwargs,
):
    task_tracker = OrderingTracking()
    if tracking_key and model_index:
        task_tracker.decr(tracking_key=tracking_key, model_id=model_index, amount=score)


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
    if "reply_id" not in kwargs:
        kwargs.update(
            reply_id=self.request.id,
        )
    return AIResourceManager.get_instance().run(*args, **kwargs)


@celery_app.task(
    name=TaskNames.MODEL_INVOKE.value,
    autoretry_for=(SoftTimeLimitExceeded, TimeLimitExceeded),
    max_retries=TaskRetryConfigs.DEFAULT_RETRY_TIMES,
    default_retry_delay=TaskRetryConfigs.STANDARD_RETRY_COUNTDOWN,
)
def model_invoke(
    model_type: str = None,
    model_index: str = None,
    **kwargs,
):
    model_class = RedisAIModel.get_submodels_map().get(model_type)
    if model_class is not None:
        instance_config = kwargs.get("instance_config", {})

        tracking_key = RedisAIModelKeyGen.gen_key(model_type)
        start_time = datetime.now()

        model = model_class.get_instance(
            model_prefix=f"{model_class.class_prefix()}:{model_index}",
            **instance_config
        )
        result = model.process(client=SyncRedisClient.get_instance().client, **kwargs)

        task_tracker = OrderingTracking()
        task_kwargs = dict(
            tracking_key=tracking_key,
            model_id=model_index,
            amount=(datetime.now() - start_time).total_seconds()
        )
        task_tracker.incr(
            **task_kwargs
        )
        model_response_time_order.apply_async(
            kwargs=task_kwargs,
            countdown=task_tracker.time_window
        )

        task_channel = kwargs.get("reply_id", "all")
        SyncRedisClient.get_instance().client.publish(str(task_channel), message=pickle.dumps(result))
