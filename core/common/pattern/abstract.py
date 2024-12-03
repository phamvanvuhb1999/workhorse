from abc import ABC


class ImplementationError(Exception):
    pass


class NotImplementRaiser(ABC):
    def _raise_not_implemented(self, message="This method must be implemented for using in subclass"):
        raise ImplementationError(message)
