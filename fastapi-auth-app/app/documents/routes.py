from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.auth.jwt_handler import decode_access_token

router = APIRouter(prefix="/documents", tags=["documents"])
bearer_scheme = HTTPBearer()

@router.post("/verify")
async def verify_document(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    token = credentials.credentials
    user = decode_access_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Continue with document processing...
    return {"message": "Token valid, file received"}
