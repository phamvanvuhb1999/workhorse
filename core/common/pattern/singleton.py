from abc import ABC


class Singleton(ABC):
    instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = cls(*args, **kwargs)
        return cls.instance
