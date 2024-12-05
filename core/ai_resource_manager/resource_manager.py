import re
import uuid
from typing import Any

from redis import Redis

from configs.common import settings

from core.common.pattern.singleton import Singleton
from core.ai_resource_manager import RedisAIModel
from core.exceptions.resource_manager import ModelNotAvailable


class AIResourceManager(Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        host = kwargs.get("host", settings.REDIS_HOST)
        port = kwargs.get("port", settings.REDIS_PORT)
        db = kwargs.get("db", settings.REDIS_DB)

        self.redis_client = Redis(host=host, port=port, db=db)

    def get_available_model(self, model_prefix: str):
        for prefix, item in RedisAIModel.get_submodels_map().items():
            if prefix != model_prefix:
                continue

            pattern = f"{prefix}:*"
            scan_results: list = []
            cursor: Any = 0
            while True:
                cursor, matching_results = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern
                )

                if matching_results:
                    scan_results = scan_results + [item.decode("utf-8") for item in matching_results]

                if not cursor:
                    break

            model_pattern = rf'{prefix}:([0-9a-fA-F/\-]{{36}}):.*'
            all_model_indexes = set([
                j
                for i in scan_results
                for j in re.findall(model_pattern, i)
            ])

            model_pattern = f"{prefix}:([0-9a-fA-F\-]{{36}}):busy"
            busy_model_indexes = set([
                j
                for i in scan_results
                for j in re.findall(model_pattern, i)
            ])
            return all_model_indexes - busy_model_indexes

    def assign_task(self, model_type, model_index, task_kwargs):
        from core.queueing.tasks.redis_ai import model_invoke

        kwargs = {
            **task_kwargs,
            "model_type": model_type,
            "model_index": model_index,
        }
        model_invoke.apply(
            kwargs=kwargs
        )

    def run(self, model_type: str, *args, **kwargs):
        available_models = self.get_available_model(model_type)
        if not available_models:
            raise ModelNotAvailable()

        model_index: str | None = None
        block_key: str | None = None

        block_identify: str = str(uuid.uuid4())
        for index in available_models:
            block_key: str = f"{model_type}:{index}:busy"
            if self.redis_client.set(
                name=block_key,
                value=block_identify,
                nx=True,
                ex=settings.MODEL_INFERENCE_TIMEOUT,
            ):
                model_index = index
                break

        if not model_index:
            raise ModelNotAvailable()

        self.assign_task(
            model_type=model_type,
            model_index=model_index,
            task_kwargs={
                **kwargs,
                "block_key": block_key,
                "block_identify": block_identify,
            },
        )

        return block_key
