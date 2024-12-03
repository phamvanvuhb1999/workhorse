class KeyGenerator:
    prefix: str = "workhorse"
    collection_key: str = ""

    @classmethod
    def gen_collection_key(cls, *args, **kwargs):
        return f"{cls.prefix}:{cls.collection_key}".format(*args, **kwargs)

    @classmethod
    def gen_key(cls, identifier, *args, **kwargs):
        return f"{cls.gen_collection_key(*args, **kwargs)}:{identifier}"


class RDSTableKeyGen(KeyGenerator):
    collection_key = "tables:{table_name}"


class TokenExpiringKeyGen(KeyGenerator):
    collection_key = "expired_token"


class QueueTrackingKeyGen(KeyGenerator):
    collection_key = "queue_tracking"
