from typing import Optional, Any

from pydantic import BaseModel
from urllib.parse import urlparse


class RedisConfig(BaseModel):
    host: Optional[str] = "localhost"
    port: Optional[str | int] = 6379
    db: Optional[str | int] = "0"
    url: Optional[str] = ""
    password: Optional[str] = None

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        url = data.get("url")
        if url:
            parsed_url = urlparse(url)
            data.update(
                host=parsed_url.hostname,
                port=parsed_url.port,
                password=parsed_url.password,
                db=int(parsed_url.path.strip('/')) if parsed_url.path else 0,
            )
        else:
            password = data.get("password")
            data.update(
                url=f"redis://{f'{password}@' if password else ''}{{host}}:{{port}}/{{db}}".format(**data)
            )
        return data