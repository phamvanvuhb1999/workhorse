import json
import logging
import pickle

logger = logging.getLogger(__name__)


class BaseSerializer:
    @classmethod
    def dumps(cls, value):
        return value

    @classmethod
    def loads(cls, value):
        return value


class JsonSerializer(BaseSerializer):
    @classmethod
    def dumps(cls, value):
        try:
            return json.dumps(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"JsonSerializer failed to decode, {e}")

    @classmethod
    def loads(cls, value):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"JsonSerializer failed to encode, {e}")


class PickleSerializer(BaseSerializer):
    @classmethod
    def dumps(cls, value):
        try:
            return pickle.dumps(value, 0)
        except (pickle.PicklingError, AttributeError, TypeError, RecursionError) as e:
            logger.error(f"PickleSerializer failed to encode, {e}")

    @classmethod
    def loads(cls, value):
        try:
            return pickle.loads(value)
        except (pickle.PicklingError, AttributeError, TypeError, RecursionError) as e:
            logger.error(f"PickleSerializer failed to decode, {e}")
