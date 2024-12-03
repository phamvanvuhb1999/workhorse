import redis

from configs.common import settings
from core.common.pattern.abstract import NotImplementRaiser
from core.common.pattern.singleton import Singleton


class BaseRedisClient(Singleton, NotImplementRaiser):
    instance_class: callable = redis.Redis

    def __init__(
        self,
        host: str = settings.REDIS_HOST,
        port: int = settings.REDIS_PORT,
        db: int = settings.REDIS_DB,
        password: str | None = settings.REDIS_PASSWORD,
        decode_responses: bool = False,
        url: str | None = None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        connect_params = dict(
            decode_responses=decode_responses,
        )
        if password:
            connect_params.update(password=password)

        if not url and settings.TESTING:
            url = settings.REDIS_URL
        if url:
            connect_params.update(
                url=url,
            )
            self.client = getattr(self.instance_class, "from_url")(**connect_params)
        else:
            connect_params.update(
                host=host,
                port=port,
                db=db,
            )
            self.client = self.instance_class(**connect_params)

    def pubsub(self):
        """Get pubsub instance."""
        return self.client.pubsub()

    def pipeline(self):
        """Get a pipeline object for batch operations."""
        return self.client.pipeline()

    async def async_set(self, *args, **kwargs):
        self._raise_not_implemented()

    def set(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_get(self, *args, **kwargs):
        self._raise_not_implemented()

    def get(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_delete(self, *args, **kwargs):
        self._raise_not_implemented()

    def delete(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_exists(self, *args, **kwargs):
        self._raise_not_implemented()

    def exists(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_keys(self, *args, **kwargs):
        self._raise_not_implemented()

    def keys(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_hset(self, *args, **kwargs):
        self._raise_not_implemented()

    def hset(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_hget(self, *args, **kwargs):
        self._raise_not_implemented()

    def hget(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_hgetall(self, *args, **kwargs):
        self._raise_not_implemented()

    def hgetall(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_publish(self, *args, **kwargs):
        self._raise_not_implemented()

    def publish(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_expire(self, *args, **kwargs):
        self._raise_not_implemented()

    def expire(self, *args, **kwargs):
        self._raise_not_implemented()

    def zadd(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_zadd(self, *args, **kwargs):
        self._raise_not_implemented()

    def zremrangebyscore(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_zremrangebyscore(self, *args, **kwargs):
        self._raise_not_implemented()

    def zcard(self, *args, **kwargs):
        self._raise_not_implemented()

    def zcount(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_zcount(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_zcard(self, *args, **kwargs):
        self._raise_not_implemented()

    def llen(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_llen(self, *args, **kwargs):
        self._raise_not_implemented()

    def lpush(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_lpush(self, *args, **kwargs):
        self._raise_not_implemented()

    def lpop(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_lpop(self, *args, **kwargs):
        self._raise_not_implemented()

    def flushdb(self, *args, **kwargs):
        self._raise_not_implemented()

    async def async_flushdb(self, *args, **kwargs):
        self._raise_not_implemented()
