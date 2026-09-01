from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from retail_search.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    SearchRequest,
    SearchResponse,
)
from retail_search.api.service import SearchService
from retail_search.artifacts.manager import ArtifactManager

LOGGER = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(artifact_manager: ArtifactManager | None = None) -> FastAPI:
    manager = artifact_manager or ArtifactManager(
        Path(os.getenv("RETAIL_ARTIFACT_DIR", "artifacts")),
        Path(os.getenv("RETAIL_MODEL_POINTER", "artifacts/promoted.json")),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            application.state.search_service = SearchService(manager.load())
            application.state.load_error = None
        except Exception as error:  # health endpoint reports the actionable startup failure
            LOGGER.exception("Unable to load promoted search artifacts")
            application.state.search_service = None
            application.state.load_error = str(error)
        yield

    application = FastAPI(
        title="Retail Search & Ranking",
        version="0.1.0",
        description="Two-stage retrieval and ML reranking with an Amazon ESCI proof adapter",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")

    def service(request: Request) -> SearchService:
        search_service = request.app.state.search_service
        if search_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"Promoted artifacts unavailable: {request.app.state.load_error}",
            )
        return search_service

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def landing_page(request: Request) -> HTMLResponse:
        search_service = service(request)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"model_info": search_service.model_info()},
        )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        search_service = request.app.state.search_service
        if search_service is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error)
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            index_loaded=True,
            model_version=search_service.version,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        return ModelInfoResponse(**service(request).model_info())

    @application.post("/search", response_model=SearchResponse)
    async def search(payload: SearchRequest, request: Request) -> SearchResponse:
        return SearchResponse(**service(request).search(payload.query, payload.top_k))

    @application.get("/demo/queries")
    async def demo_queries(request: Request) -> list[dict[str, str]]:
        return service(request).curated_queries()

    @application.get("/compare/{query_id}")
    async def compare(query_id: str, request: Request) -> dict:
        try:
            return service(request).comparison(query_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Curated benchmark query not found") from error

    return application


app = create_app()


def run() -> None:
    uvicorn.run("retail_search.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
