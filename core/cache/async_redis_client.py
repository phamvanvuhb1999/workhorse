from typing import Optional

from redis.asyncio import Redis

from core.cache.base import BaseRedisClient
from core.cache.serializers import BaseSerializer


class AsyncRedisClient(BaseRedisClient):
    instance_class: callable = Redis

    async def async_flushdb(self, *args, **kwargs):
        return await self.client.flushdb(*args, **kwargs)

    async def async_set(self, key: str, value: str, serializer_class=BaseSerializer, **kwargs):
        value = serializer_class.dumps(value)
        return await self.client.set(key, value, **kwargs)

    async def async_get(self, key: str, serializer_class=BaseSerializer, **kwargs) -> Optional[str]:
        value = await self.client.get(key, **kwargs)

        try:
            obj = serializer_class.loads(value) if value else value
        except Exception:
            obj = value
        return obj

    async def async_delete(self, key: str, **kwargs):
        return await self.client.delete(key, **kwargs)

    async def async_exists(self, key: str, **kwargs) -> bool:
        return await self.client.exists(key, **kwargs) > 0

    async def async_keys(self, pattern: str = '*', **kwargs) -> list:
        return await self.client.keys(pattern, **kwargs)

    async def async_hset(self, name: str, mapping: dict, serializer=BaseSerializer, **kwargs):
        mapping_data: dict = {
            key: serializer.dumps(value) if value else value
            for key, value in mapping.items()
        }
        return await self.client.hset(name, mapping=mapping_data, **kwargs)

    async def async_hget(self, name: str, key: str, serializer=BaseSerializer, **kwargs) -> Optional[str]:
        value = await self.client.hget(name, key, **kwargs)
        return serializer.loads(value) if value else value

    async def async_hgetall(self, name: str, serializer=BaseSerializer, **kwargs):
        data: dict = await self.client.hgetall(name, **kwargs)
        return {
            key.decode("utf-8"): serializer.loads(value) if value else value
            for key, value in data.items()
        }

    async def async_publish(self, channel, message, **kwargs):
        return await self.client.publish(channel, message, **kwargs)

    async def async_expire(self, key, ex_time: int = 0, **kwargs):
        return await self.client.expire(key, ex_time, **kwargs)

    async def async_zadd(self, key, item: str, score: float, *args, **kwargs):
        return await self.client.zadd(key, {item: score}, *args, **kwargs)

    async def async_zremrangebyscore(self, key: str, min_score: str = "-inf", max_score: str = "+inf"):
        return await self.client.zremrangebyscore(key, min=min_score, max=max_score)

    async def async_zcard(
        self,
        key,
        with_validate: bool = True,
        min_score: str = "-inf",
        max_score: str = "+inf",
        *args, **kwargs,
    ):
        if with_validate:
            await self.async_zremrangebyscore(key, min_score=min_score, max_score=max_score)
        return await self.client.zcard(key)

    async def async_zcount(
        self,
        key,
        min_score: str = "-inf",
        max_score: str = "+inf",
        *args, **kwargs,
    ):
        return await self.client.zcount(key, min_score, max_score)

    async def async_llen(self, key, *args, **kwargs):
        return await self.client.llen(key, *args, **kwargs)

    async def async_lpush(self, key, *args, **kwargs):
        return await self.client.lpush(key, *args, **kwargs)

    async def async_lpop(self, key, *args, **kwargs):
        return await self.client.lpop(key, *args, **kwargs)

    async def exists(self, key, *args, **kwargs):
        return await self.client.exists(key, **kwargs)
