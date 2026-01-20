from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from app import models, schemas
from app.database import get_db
from app.auth_utils import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
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
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    access_token = create_access_token({"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# ---------------- LOGIN ----------------
@router.post("/login", response_model=schemas.Token)
def login(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token({"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# ---------------- Protected Route ----------------
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None or payload.get("sub") is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return payload.get("sub")

@router.get("/me")
def read_current_user(current_user: str = Depends(get_current_user)):
    return {"email": current_user}
