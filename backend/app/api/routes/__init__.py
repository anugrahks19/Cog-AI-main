from fastapi import APIRouter

from . import assessments, users

api_router = APIRouter(prefix="/api")

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
