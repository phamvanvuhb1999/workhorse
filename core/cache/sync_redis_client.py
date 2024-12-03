import redis

from core.cache.base import BaseRedisClient
from core.cache.serializers import BaseSerializer
from core.cache.serializers import PickleSerializer


class SyncRedisClient(BaseRedisClient):
    instance_class: callable = redis.Redis

    def keys(self, pattern: str = '*', **kwargs) -> list:
        return self.client.keys(pattern, **kwargs)

    def set(self, key, value, serializer_class=BaseSerializer, **kwargs):
        """Set a value in Redis, serialized with pickle."""
        serialized_value = serializer_class.dumps(value)
        return self.client.set(key, serialized_value, **kwargs)

    def get(self, key, serializer_class=BaseSerializer, **kwargs):
        """Get a value from Redis, deserialized with pickle."""
        serialized_value = self.client.get(key, **kwargs)

        try:
            obj = serializer_class.loads(serialized_value) if serialized_value else serialized_value
        except Exception:
            obj = serialized_value
        return obj

    def hset(self, name: str, mapping: dict, serializer_class=BaseSerializer, **kwargs):
        """Set a value in a hash in Redis, serialized with pickle."""
        mapping_data: dict = {
            key: serializer_class.dumps(value) if value else value
            for key, value in mapping.items()
        }
        return self.client.hset(name, mapping=mapping_data, **kwargs)

    def hget(self, name, key, serializer_class=BaseSerializer, **kwargs):
        """Get a value from a hash in Redis, deserialized with pickle."""
        value = self.client.hget(name, key, **kwargs)
        if value is not None:
            value = serializer_class.loads(value)
        return value

    def hexists(self, name, key, **kwargs):
        """Check if a field exists in a hash in Redis."""
        return self.client.hexists(name, key, **kwargs)

    def hkeys(self, name, **kwargs):
        """Get all the fields in a hash in Redis."""
        return self.client.hkeys(name, **kwargs)

    def hvals(self, name, serializer_class=PickleSerializer, **kwargs):
        """Get all the values in a hash in Redis, deserialized with pickle."""
        serialized_values = self.client.hvals(name, **kwargs)
        return [serializer_class.loads(value) for value in serialized_values]

    def hgetall(self, name, serializer_class=BaseSerializer, **kwargs):
        """Get all the fields and values in a hash in Redis, deserialized with pickle."""
        serialized_dict = self.client.hgetall(name, **kwargs)
        return {key.decode("utf-8"): serializer_class.loads(value) for key, value in serialized_dict.items()}

    def hlen(self, name, **kwargs):
        """Get the number of fields in a hash in Redis."""
        return self.client.hlen(name, **kwargs)

    def sadd(self, key, *values, serializer_class=BaseSerializer, **kwargs):
        """Add one or more values to a set in Redis, serialized with pickle."""
        serialized_values = [serializer_class.dumps(value) for value in values]
        return self.client.sadd(key, *serialized_values, **kwargs)

    def smembers(self, key, serializer_class=BaseSerializer, **kwargs):
        """Get all members of a set in Redis, deserialized with pickle."""
        serialized_values = self.client.smembers(key, **kwargs) or []
        return {serializer_class.loads(value) for value in serialized_values}

    def srem(self, key, *values, serializer_class=BaseSerializer, **kwargs):
        """Remove one or more values from a set in Redis."""
        serialized_values = [serializer_class.dumps(value) for value in values]
        return self.client.srem(key, *serialized_values, **kwargs)

    def publish(self, channel: str, message: str, *args, **kwargs):
        return self.client.publish(channel, message, **kwargs)

    def delete(self, key, *args, **kwargs):
        return self.client.delete(key, **kwargs)

    def zadd(self, key, item: str, score: str, *args, **kwargs):
        return self.client.zadd(key, {item: score}, *args, **kwargs)

    def zremrangebyscore(self, key: str, min_score: str = "-inf", max_score: str = "+inf", *args, **kwargs):
        return self.client.zremrangebyscore(key, min=min_score, max=max_score)

    def zcard(
        self,
        key,
        with_validate: bool = True,
        min_score: str = "-inf",
        max_score: str = "+inf",
        *args, **kwargs,
    ):
        if with_validate:
            self.zremrangebyscore(key, min_score=min_score, max_score=max_score)
        return self.client.zcard(key)

    def zcount(
        self,
        key,
        min_score: str = "-inf",
        max_score: str = "+inf",
        *args, **kwargs,
    ):
        return self.client.zcount(key, min_score, max_score)

    def llen(self, key, *args, **kwargs):
        return self.client.llen(key, *args, **kwargs)

    def lpush(self, key, *args, **kwargs):
        return self.client.lpush(key, *args, **kwargs)

    def lpop(self, key, *args, **kwargs):
        return self.client.lpop(key, *args, **kwargs)

    def flushdb(self, *args, **kwargs):
        return self.client.flushdb(*args, **kwargs)

    def expire(self, key, ex_time: int = 0, **kwargs):
        return self.client.expire(key, ex_time, **kwargs)

    def exists(self, key, *args, **kwargs):
        return self.client.exists(key, **kwargs)
