from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import User
from .hashing import hash_password, verify_password
from .jwt_handler import create_access_token
from .schemas import UserCreate, UserLogin, Token

router = APIRouter(prefix="/auth", tags=["auth"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# Signup
# -----------------------------
@router.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = User(
        email=user.email,
        hashed_password=hash_password(user.password),
        full_name=user.full_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.email})
    return {"access_token": token, "token_type": "bearer"}

# -----------------------------
# Login
# -----------------------------
@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

@router.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        new_user = User(
            email=user.email,
            hashed_password=hash_password(user.password),
            full_name=user.full_name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        token = create_access_token({"sub": new_user.email})
        return {"access_token": token, "token_type": "bearer"}

    except Exception as e:
        print("ERROR:", e)  # 👈 prints real reason in terminal
        raise HTTPException(status_code=500, detail=str(e))

