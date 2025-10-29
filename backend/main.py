from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from api.routes import router as api_router
import uvicorn
from pathlib import Path

app = FastAPI(title="Sigma108 Backend Placeholder")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding='utf-8'))
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Frontend application not found. Check build.")

@app.get("/{full_path:path}")
async def serve_react_app_fallback(full_path: str):
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="API endpoint not found")

    file_path = STATIC_DIR / full_path
    if file_path.exists() and file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    else:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding='utf-8'))
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Frontend application not found. Check build.")
