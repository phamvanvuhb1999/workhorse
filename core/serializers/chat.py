from pydantic import BaseModel


class ChatData(BaseModel):
    message: str
