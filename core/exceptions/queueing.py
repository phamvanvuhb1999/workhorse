from starlette import status

from core.exceptions.base import BaseHTTPException

class MessageOrderingException(BaseHTTPException):
    detail = "Task executed ordering incorrect, retry later."
    status_code = status.HTTP_400_BAD_REQUEST
