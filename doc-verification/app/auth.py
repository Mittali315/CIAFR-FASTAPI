from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from app import models, schemas
from app.database import get_db
from app.auth_utils import hash_password, verify_password, create_access_token, decode_access_token

# Router instance
router = APIRouter(prefix="/auth", tags=["auth"])

# OAuth2 scheme for protected routes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ---------------- SIGNUP ----------------
@router.post("/signup", response_model=schemas.Token)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
        is_active=True,
        is_superuser=False
    )