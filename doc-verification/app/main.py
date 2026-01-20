from fastapi import FastAPI
from app.database import Base, engine
from app.auth.routes import router as auth_router

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="Document Verification Auth API")

# Include auth routes
app.include_router(auth_router)
