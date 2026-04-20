"""FastAPI entrypoint for the LLM-Based Algorithm Visualizer backend."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes.debug import router as debug_router
from app.routes.solve import router as solve_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="LLM-Based Algorithm Visualizer API",
    version="1.0.0",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app.include_router(solve_router)
app.include_router(debug_router)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the minimal visualization UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health endpoint for service monitoring."""
    return {"status": "ok"}
