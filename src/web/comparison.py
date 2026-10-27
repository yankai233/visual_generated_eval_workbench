from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.data.result_catalog import ResultCatalog
from src.pipelines.registry import discover_pipelines
from src.utils.paths import TEMPLATE_ROOT


class ComparisonColumn(BaseModel):
    pipeline: str = Field(min_length=1, max_length=128)
    config_name: str | None = None
    run_id: str | None = None


class ComparisonView(BaseModel):
    source: str
    columns: list[ComparisonColumn] = Field(min_length=1, max_length=8)
    sample_ids: list[str] = Field(default_factory=list, max_length=10000)
    query: str = Field(default="", max_length=1000)
    only_common: bool = False
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=50)


class ComparisonGroup(BaseModel):
    result_ids: list[str] = Field(min_length=1, max_length=500)
    name: str | None = Field(default=None, max_length=120)


def comparison_router(catalog: ResultCatalog) -> APIRouter:
    router = APIRouter()

    @router.get("/compare", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(TEMPLATE_ROOT / "compare.html")

    @router.get("/api/comparison/catalog")
    def index() -> dict:
        payload = catalog.catalog()
        payload["pipeline_labels"] = {
            name: cls.display_name or name for name, cls in discover_pipelines().items()
        }
        return payload

    @router.get("/api/comparison/samples")
    def samples(source: str) -> dict:
        return {"samples": catalog.samples(source)}

    @router.get("/api/comparison/records")
    def records(source: str = "users") -> dict:
        return {"records": catalog.records(source)}

    @router.post("/api/comparison/view")
    def view(payload: ComparisonView) -> dict:
        return catalog.view(payload.model_dump())

    @router.post("/api/comparison/group")
    def group(payload: ComparisonGroup) -> dict:
        try:
            catalog.set_group(payload.result_ids, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    return router
