import re
import uuid
from typing import Any

from redis import Redis

from configs.common import settings

from core.common.pattern.singleton import Singleton
from core.ai_resource_manager import RedisAIModel
from core.exceptions.resource_manager import ModelNotAvailable
from core.serializers.redis_config import RedisConfig


class AIResourceManager(Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        host = kwargs.get("host", settings.REDIS_HOST)
        port = kwargs.get("port", settings.REDIS_PORT)
        db = kwargs.get("db", settings.REDIS_DB)
        redis_ai_hosts = kwargs.get("redis_ai_host", settings.REDIS_AI_HOSTS or [settings.REDIS_URL])

        self.redis_ai_hosts = redis_ai_hosts
        self.redis_client = Redis(host=host, port=port, db=db)

    def get_available_model(self, model_prefix: str):
        scan_results: list = []
        for prefix, item in RedisAIModel.get_submodels_map().items():
            if prefix != model_prefix:
                continue
            pattern = f"{prefix}:*"
            for redis_url in self.redis_ai_hosts:
                client = Redis.from_url(url=redis_url)

                results: list = []
                cursor: Any = 0
                while True:
                    cursor, matching_results = client.scan(
                        cursor=cursor,
                        match=pattern
                    )

                    if matching_results:
                        results = results + [item.decode("utf-8") for item in matching_results]

                    if not cursor:
                        break

                model_pattern = rf'{prefix}:([0-9a-fA-F/\-]{{36}}):.*'
                all_model_indexes = [
                    j
                    for i in results
                    for j in re.findall(model_pattern, i)
                ]

                scan_results.append({
                    "config": RedisConfig(url=redis_url).model_dump(),
                    "available_indexes": all_model_indexes,
                })
        return scan_results

    def assign_task(self, model_type, model_index, task_kwargs):
        from core.queueing.tasks.redis_ai import model_invoke

        kwargs = {
            **task_kwargs,
            "model_type": model_type,
            "model_index": model_index,
        }
        model_invoke.apply_async(
            kwargs=kwargs
        )

    def run(self, model_type: str, lock_id: str, *args, **kwargs):
        scan_results = self.get_available_model(model_type)
        if all(not item.get("available_indexes") for item in scan_results):
            raise ModelNotAvailable()

        def get_model_index(scan_data):
            identify: str = str(uuid.uuid4())
            for item in scan_data:
                for index in item.get("available_indexes"):
                    key = f"{model_type}:{index}:{lock_id}:busy"
                    if settings.REDIS_AI_INSTANCE_LOCK and not self.redis_client.set(
                        name=key,
                        value=identify,
                        nx=True,
                        ex=settings.MODEL_INFERENCE_TIMEOUT
                    ):
                        continue
                    else:
                        return index, item.get("config"), key, identify
            return None, None, None, None

        model_index, instance_config, block_key, block_identify = get_model_index(scan_results)
        if not model_index:
            raise ModelNotAvailable()

        self.assign_task(
            model_type=model_type,
            model_index=model_index,
            task_kwargs={
                **kwargs,
                "lock_id": lock_id,
                "block_key": block_key,
                "block_identify": block_identify,
                "instance_config": instance_config,
            },
        )

        return block_key
