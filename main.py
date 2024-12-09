import asyncio
import os
import uuid
from typing import Annotated

from fastapi import Request, UploadFile, File

from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from core.ai_resource_manager import RedisAIModel
from core.ai_resource_manager import PaddleDetectorRedisModel
from core.ai_resource_manager import T5RedisModel
from core.queueing.constants import QueueNames
from core.serializers.chat import ChatData
from core.utils import T5InferenceHelper
from core.queueing.tasks.redis_ai import push_to_source_manager


from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.exception_handlers import http_exception_handler
from starlette.middleware.cors import CORSMiddleware

from configs.common import settings
from core.exceptions.base import BaseHTTPException
from core.exceptions.exception_handlers import unexpected_exception_handler

load_dotenv()


app = FastAPI(
    swagger_ui_parameters={"syntaxHighlight": False},
    title="WHS",
    description="FastAPI Project for WHS",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")

# Add middlewares
app.add_middleware(
    CORSMiddleware,  # noqa
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGINS_REGEX,
    allow_credentials=True,
    allow_methods=("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"),
    allow_headers=settings.CORS_HEADERS,
)

# Base exception handler
app.add_exception_handler(HTTPException, http_exception_handler)  # noqa
app.add_exception_handler(BaseHTTPException, unexpected_exception_handler)  # noqa

@app.get('/', status_code=status.HTTP_200_OK)
async def ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post('/chat', status_code=status.HTTP_200_OK)
async def chat(data: ChatData):
    try:
        indices = T5InferenceHelper.get_batched_indices(data.message)
    except KeyError:
        reply = "I did not understand your language!!, check the spelling perhaps"
    else:
        push_to_source_manager.apply_async(
            kwargs={
                "indices": indices,
                "model_type": T5RedisModel.class_prefix(),
            },
            routing_key=QueueNames.WAITING,
            queue=QueueNames.WAITING
        )
        reply = await RedisAIModel.wait_for_response(key="all")

    return {"reply": reply}


@app.post('/ocr', status_code=status.HTTP_200_OK)
async def ocr(
    front: Annotated[UploadFile, File(...)],
    back: Annotated[UploadFile, File(...)]
):
    await front.seek(0)
    await back.seek(0)

    front_image = await front.read()
    back_image = await back.read()
    lock_id = uuid.uuid4() if settings.REDIS_AI_SESSION_LOCK else ""

    front_task_id = push_to_source_manager.apply_async(
        kwargs={
            "image": front_image,
            "lock_id": lock_id,
            "model_type": PaddleDetectorRedisModel.class_prefix(),
        },
        routing_key=QueueNames.WAITING,
        queue=QueueNames.WAITING
    )
    back_task_id = push_to_source_manager.apply_async(
        kwargs={
            "image": back_image,
            "lock_id": lock_id,
            "model_type": PaddleDetectorRedisModel.class_prefix(),
        },
        routing_key=QueueNames.WAITING,
        queue=QueueNames.WAITING
    )
    await asyncio.gather(
        RedisAIModel.wait_for_response(key=str(front_task_id)),
        RedisAIModel.wait_for_response(key=str(back_task_id))
    )
    return {"is_finished": True}


@app.get('/{filepath:path}', status_code=status.HTTP_200_OK)
async def ui_components(filepath: str):
    if not filepath:
        return await ui()

    else:
        filepath = os.path.join("static", filepath)
        return FileResponse(filepath, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=settings.PORT)
