import logging

from fastapi import FastAPI

from dub.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Dub - AI Video Dubbing", version="0.1.0")

app.include_router(router)
