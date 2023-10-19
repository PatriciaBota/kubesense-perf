from fastapi.exceptions import HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from src.core.logger import logger


class Errors(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        data: Dict[str, Any] = {},
        headers: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code, HTTPDetail(message=message, data=data).dict(), headers
        )
        self.__log(message, data, status_code)

    def __log(self, message: str, data: Dict[str, Any], status_code: int):
        logger.error(message, data=data, status_code=status_code)


class HTTPDetail(BaseModel):
    message: str
    data: Dict[str, Any]
