import contextlib
import json
import pickle
import re
import uuid
from typing import Any


def save_convert(
    value: Any,
    target_type: Any,
    accepted_none: bool = True,
    allow_true_type: bool = False,
):
    if not callable(target_type):
        raise Exception("Target type must be callable.")

    result: Any = None

    converters = [target_type]

    if allow_true_type:
        converters = converters + [json.loads, pickle.loads, json.load, pickle.load]

    for converter in converters:
        if result is None:
            with contextlib.suppress(Exception):
                result = converter(value)
        else:
            break

    if result is None and not accepted_none:
        result = target_type()

    return result


class UploadFileKeyGen:
    @classmethod
    def format_key(cls, string: str, key_hyp: str):
        return re.sub(r'[^a-zA-Z0-9.]', key_hyp, str(string).lower())

    @classmethod
    def gen_key(cls, file_name: str, prefix: str = "uploads", file_id: str = None, key_hyp: str = "_"):
        formatted_filename: str = cls.format_key(file_name, key_hyp)
        return f"{prefix}/{file_id or uuid.uuid4()}_{formatted_filename}"
