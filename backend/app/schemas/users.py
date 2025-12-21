from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from app.models import User


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    age: int = Field(..., ge=18, le=120)
    gender: int = Field(..., ge=0, le=1)
    education: int = Field(..., ge=0, le=3)
    # New Health Fields
    family_history: int = Field(0, ge=0, le=1)
    diabetes: int = Field(0, ge=0, le=1)
    hypertension: int = Field(0, ge=0, le=1)
    depression: int = Field(0, ge=0, le=1)
    head_injury: int = Field(0, ge=0, le=1)
    sleep_quality: int = Field(7, ge=0, le=10)
    physical_activity: int = Field(5, ge=0, le=10)
    smoking: int = Field(0, ge=0, le=1)
    alcohol_consumption: float = Field(0.0, ge=0, le=50) # Drinks/week
    diet_quality: float = Field(5.0, ge=0, le=10)
    height: float = Field(None, gt=0) # cm
    weight: float = Field(None, gt=0) # kg
    
    language: str = Field(..., min_length=2, max_length=16)
    consent: bool


class UserOut(BaseModel):
    id: str
    name: str
    age: int
    gender: int
    education: int
    
    family_history: int = 0
    diabetes: int = 0
    hypertension: int = 0
    depression: int = 0
    head_injury: int = 0
    sleep_quality: int = 7
    physical_activity: int = 5
    smoking: int = 0
    alcohol_consumption: float = 0.0
    diet_quality: float = 5.0
    height: float | None = None
    weight: float | None = None

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
