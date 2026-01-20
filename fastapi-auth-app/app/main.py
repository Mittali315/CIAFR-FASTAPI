from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from app.auth.routes import router as auth_router
from app.documents.routes import router as documents_router
from fastapi.security import HTTPBearer
app = FastAPI(title="Document Verification API")

app.include_router(auth_router)
app.include_router(documents_router)

# Security scheme for Bearer token
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

bearer_scheme = HTTPBearer()
