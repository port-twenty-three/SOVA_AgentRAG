#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.app.api.v1 import chat, delete, health, upload
from service.utils.config import settings
from service.utils.log import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Действия при запуске приложения
    logger.info(r"""
 ____   ___  _   _    _
/ ___| / _ \| | | |  / \
\___ \| | | | | | | / _ \
 ___) | |_| |\ V / / ___ \
|____/ \___/  \_/ /_/   \_\

  developer alakei1
    """)
    yield
    # Действия при остановке приложения
    logger.info("Shutting down RAG Service...")


app = FastAPI(
    title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan
)

app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(delete.router, prefix="/api/v1", tags=["delete"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.main:app", host="0.0.0.0", port=8000, reload=True)