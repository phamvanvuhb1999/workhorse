from fastapi.encoders import jsonable_encoder
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.exceptions.base import BaseHTTPException


def unexpected_exception_handler(request: Request, exc: BaseHTTPException) -> JSONResponse:  # noqa
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": jsonable_encoder(exc.detail)},
    )
