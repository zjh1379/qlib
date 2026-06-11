from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.charts.router import router as charts_router
from app.core.config import Settings
from app.core.db import dispose_db_singletons, init_db_singletons
from app.core.exceptions import BusinessError
from app.core.logging import configure_logging, get_logger
from app.core.qlib_adapter import init_qlib_once
from app.data.router import instruments_router as data_instruments_router
from app.data.router import router as data_router
from app.evaluation.router import router as evaluation_router
from app.inference.router import router as inference_router, internal_router as inference_internal_router
from app.analysis.router import router as analysis_router, internal_router as analysis_internal_router
from app.models.router import router as models_router
from app.ops.router import router as ops_router
from app.portfolio.router import router as portfolio_router
from app.scheduling.router import router as scheduling_router, set_manager
from app.scheduling.service import SchedulerManager, make_subprocess_retrain_job


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

    # Scheduler manager — the real retrain runs `production.rolling_train run-once`
    # in a *subprocess* so the FastAPI event loop is never blocked by the 1.5–4
    # hours of CPU/GPU work. (The subprocess script is delivered in T10 onwards;
    # before T10 lands the job will exit non-zero but never crash the API.)
    repo_root = Path(__file__).resolve().parent.parent.parent
    retrain_job = make_subprocess_retrain_job(
        python_path=settings.retrain_python_path,
        repo_root=repo_root,
    )

    manager = SchedulerManager(retrain_job)
    set_manager(manager)
    from app.core import db as _db
    if _db._session_maker is None:
        raise RuntimeError("DB singletons not initialized — init_db_singletons() must run before scheduling start")
    async with _db._session_maker() as session:
        await manager.start(session)

    log.info("app_started", port=settings.api_port)
    yield
    await manager.stop()
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
    app.include_router(data_router, prefix="/api/data", tags=["data"])
    app.include_router(data_instruments_router, prefix="/api", tags=["data"])
    app.include_router(portfolio_router, prefix="/api/portfolio", tags=["portfolio"])
    app.include_router(models_router, prefix="/api/models", tags=["models"])
    app.include_router(evaluation_router, prefix="/api/evaluation", tags=["evaluation"])
    app.include_router(inference_router)
    app.include_router(inference_internal_router)
    app.include_router(analysis_router)
    app.include_router(analysis_internal_router)
    app.include_router(scheduling_router, prefix="/api/scheduling", tags=["scheduling"])

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
            # Guard: 404 on API/docs paths that should never hit SPA fallback
            if (
                full_path.startswith("api/")
                or full_path.startswith("docs")
                or full_path.startswith("redoc")
                or full_path == "openapi.json"
            ):
                raise HTTPException(status_code=404, detail="Not Found")
            # Return index.html for SPA routing
            return FileResponse(str(index_html), media_type="text/html")

    return app


app = create_app()
