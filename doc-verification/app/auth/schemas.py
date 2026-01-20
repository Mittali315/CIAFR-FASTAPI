from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    full_name: str | None = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    email: str
    full_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
