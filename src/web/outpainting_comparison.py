from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.data.outpainting_catalog import OutpaintingResultCatalog
from src.utils.paths import TEMPLATE_ROOT


def outpainting_comparison_router(catalog: OutpaintingResultCatalog) -> APIRouter:
    router = APIRouter()

    @router.get("/outpainting-compare", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(TEMPLATE_ROOT / "outpainting-compare.html")

    @router.get("/api/outpainting-comparison/catalog")
    def index(refresh: bool = False) -> dict:
        return catalog.catalog(force=refresh)

    @router.get("/api/outpainting-comparison/view")
    def view(
        runs: list[str] = Query(default=[]),
        query: str = Query(default="", max_length=1000),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=12, ge=1, le=30),
    ) -> dict:
        return catalog.view(runs, query=query, page=page, page_size=page_size)

    @router.get("/api/outpainting-comparison/image/{run_id}/{sample_id}/{kind}/{index}")
    def image(run_id: str, sample_id: str, kind: str, index: int) -> FileResponse:
        path = catalog.image(run_id, sample_id, kind, index)
        if path is None:
            raise HTTPException(status_code=404, detail="图像不存在")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    return router
