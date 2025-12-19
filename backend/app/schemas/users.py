from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from app.models import User


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    age: int = Field(..., ge=18, le=120)
    language: str = Field(..., min_length=2, max_length=16)
    consent: bool


class UserOut(BaseModel):
    id: str
    name: str
    age: int
    language: str
    consent: bool
    created_at: datetime

    # Pydantic v2: enable ORM object parsing
    model_config = ConfigDict(from_attributes=True)


class UserResponse(BaseModel):
    user: UserOut
    access_token: str
    expires_at: datetime

    @classmethod
    def from_user(cls, user: User, access_token: str, expires_at: datetime) -> "UserResponse":
        # Pydantic v2: use model_validate for ORM objects
        return cls(user=UserOut.model_validate(user), access_token=access_token, expires_at=expires_at)
