from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.charts.router import router as charts_router
from app.core.config import Settings
from app.core.db import dispose_db_singletons, init_db_singletons
from app.core.exceptions import BusinessError
from app.core.logging import configure_logging, get_logger
from app.core.qlib_adapter import init_qlib_once
from app.ops.router import router as ops_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(json_output=True)
    log = get_logger("startup")
    init_db_singletons(settings)
    try:
        init_qlib_once(settings)
        log.info("qlib_ready")
    except Exception as e:
        log.warning("qlib_not_ready_at_boot", error=str(e))
    log.info("app_started", port=settings.api_port)
    yield
    await dispose_db_singletons()
    log.info("app_stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Qlib Companion", version="0.1.0", lifespan=lifespan)

    # CORS for local dev (Vite on :5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BusinessError)
    async def business_error_handler(_, exc: BusinessError):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    app.include_router(charts_router, prefix="/api/charts", tags=["charts"])
    app.include_router(ops_router, prefix="/api/ops", tags=["ops"])

    # Static serving of the built frontend (created in T18; tolerated if missing)
    static_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if static_dir.is_dir():
        # Mount assets directory
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # SPA catch-all: serve index.html for all unmapped routes (enables client-side routing)
        index_html = static_dir / "index.html"

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Return index.html for SPA routing
            return FileResponse(str(index_html), media_type="text/html")

    return app


app = create_app()
