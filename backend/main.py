from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(title="Sigma108 Backend Placeholder")

app.include_router(api_router)
