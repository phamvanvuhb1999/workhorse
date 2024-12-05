import asyncio
import logging
import uuid
import pickle
from datetime import datetime, timedelta

from functools import wraps
from typing import Any

import redisai as rai

from core.cache.sync_redis_client import SyncRedisClient
from core.common.pattern.abstract import NotImplementRaiser
from core.common.pattern.singleton import Singleton


logger = logging.getLogger(__name__)


def prefix_generator(keys: list = None):
    if keys is None:
        keys = ["key"]

    def caller(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def format_key(k: str, args_cp: tuple, kwargs_cp: dict):
                k_val = kwargs_cp.get(k)
                key_in_kwargs: bool = True
                if k_val is None and len(args_cp) >= 2:
                    k_val = args_cp[1]
                    key_in_kwargs = False

                if k_val is not None:
                    prefix: str = getattr(args_cp[0], 'prefix')
                    if isinstance(k_val, list):
                        k_val = [
                            f"{prefix}:{item}" for item in k_val
                        ]
                    else:
                        k_val = f"{prefix}:{k_val}"

                    if key_in_kwargs:
                        kwargs_cp = {
                            **kwargs_cp, k: k_val
                        }
                    else:
                        args_cp = [args_cp[0], k_val, *args_cp[2:]]
                return args_cp, kwargs_cp

            for key in keys:
                args, kwargs = format_key(key, args, kwargs)

            return func(*args, **kwargs)

        return wrapper

    return caller


class RedisAIModel(Singleton, NotImplementRaiser):
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        prefix: str = None,
        max_tokens: int = 100,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.max_tokens = max_tokens
        self.client = rai.Client(host=host, port=port, db=db)

        if not prefix:
            prefix = f"{self.class_prefix()}:{uuid.uuid4()}"
        self.prefix = prefix

    @classmethod
    def class_prefix(cls):
        return cls.__name__.lower()

    @classmethod
    def get_submodels_map(cls):
        return {
            item.class_prefix(): item
            for item in RedisAIModel.__subclasses__()
        }

    @prefix_generator()
    def store_model(self, key: str, backend: str, device: str, data: Any, **kwargs):
        logger.info(f"Store model with {key}")
        self.client.modelstore(key, backend=backend, device=device, data=data, **kwargs)

    @prefix_generator()
    def feed_model(self, key: str, tensor: Any, **kwargs):
        self.client.tensorset(key, tensor=tensor, **kwargs)

    @prefix_generator(keys=["key", "inputs", "outputs"])
    def execute_model(self, key: str, inputs: list, outputs: list):
        return self.client.modelexecute(key=key, inputs=inputs, outputs=outputs)

    @prefix_generator()
    def get_tensor_model(self, key: str, **kwargs):
        return self.client.tensorget(key=key, **kwargs)

    def initiate(
        self,
        **kwargs,
    ):
        self._raise_not_implemented()

    def process(
        self,
        **kwargs,
    ):
        self._raise_not_implemented()

    @classmethod
    async def wait_for_response(cls, key: str, waiting_time: int = 60):
        from core.cache.async_redis_client import AsyncRedisClient

        start_time = datetime.now()
        async_redis_client = AsyncRedisClient.get_instance()
        pubsub = async_redis_client.pubsub()
        await pubsub.subscribe(key)
        while True:
            response = await pubsub.get_message(ignore_subscribe_messages=True)
            if response:
                data = pickle.loads(response.get("data"))
                if data.get("is_finished"):
                    return data
            elif datetime.now() - start_time > timedelta(seconds=waiting_time):
                return "Time limit exceeded!"
            else:
                await asyncio.sleep(0.01)

    def release_model_lock(self, **kwargs):
        block_key = kwargs.get("block_key")
        block_identify = kwargs.get("block_identify")
        if block_key and block_identify:
            redis_client = SyncRedisClient.get_instance()

            if redis_client.get(block_key).decode("utf-8") == block_identify:
                redis_client.delete(key=block_key)
