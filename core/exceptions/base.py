from fastapi import HTTPException
from fastapi import status


class BaseHTTPException(HTTPException):
    detail = ""
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, status_code: int = 0, detail: str = "", headers: dict = None):
        super().__init__(status_code=status_code or self.status_code, detail=detail or self.detail, headers=headers)
