from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.models import User
from app.auth.schemas import UserCreate, UserLogin, Token
from app.auth.hashing import hash_password, verify_password
from app.auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- SIGNUP ----------
@router.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_user = User(
        email=user.email,
        hashed_password=hash_password(user.password),
        full_name=user.full_name,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

# ---------- LOGIN ----------
@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}
