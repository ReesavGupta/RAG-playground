from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database.postgres_db import get_db
from app.api.tickets import router as tickets_router
import uvicorn

app = FastAPI(title="Smart Customer Support System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(tickets_router, prefix="/api/tickets", tags=["tickets"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Mount static files for frontend (after API routes to avoid conflicts)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)