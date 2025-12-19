from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import api_router

settings = get_settings()

app = FastAPI(
    title="Cog.ai Mindful Scan",
    version="0.1.0",
    description="Backend services for cognitive and speech screening",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"status": "ok"}
