import re
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

    def get_available_model(self, model_type: str):
        for prefix, model_class in RedisAIModel.get_submodels_map().items():
            if prefix != model_type:
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

            model_pattern = rf'{prefix}:([0-9a-fA-F/\-]{{36}}):{model_class.model_key}'
            return [
                j
                for i in scan_results
                for j in re.findall(model_pattern, i)
            ]

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

    def run(self, model_type: str, *args, **kwargs):
        available_models = self.get_available_model(model_type)
        if not available_models:
            raise ModelNotAvailable()
        model_index = list(available_models)[0]

        self.assign_task(
            model_type=model_type,
            model_index=model_index,
            task_kwargs={
                **kwargs,
            },
        )
