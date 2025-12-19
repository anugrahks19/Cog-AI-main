from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Assessment, User
from app.schemas.auth import create_access_token
from app.schemas.users import UserCreate, UserResponse

router = APIRouter()


@router.post("", response_model=UserResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    user = User(name=payload.name, age=payload.age, language=payload.language, consent=payload.consent)
    db.add(user)
    db.commit()
    db.refresh(user)

    # Kick off assessment stub
    assessment = Assessment(user_id=user.id, language=user.language)
    db.add(assessment)
    db.commit()

    token = create_access_token(str(user.id))

    return UserResponse(user=user, access_token=token.access_token, expires_at=token.expires_at)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)) -> UserResponse:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    token = create_access_token(user.id)
    return UserResponse(user=user, access_token=token.access_token, expires_at=token.expires_at)
