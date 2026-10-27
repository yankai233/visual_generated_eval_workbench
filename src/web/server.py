from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.data.benchmark_loader import BenchmarkError, list_benchmarks, load_benchmark
from src.data.schemas import BenchmarkSelection, CreateJobRequest, SampleModel
from src.data.sample_identity import prepare_sample
from src.data.result_catalog import ResultCatalog
from src.data.outpainting_catalog import OutpaintingResultCatalog
from src.pipelines.registry import (
    discover_pipelines,
    get_pipeline,
    pipeline_discovery_errors,
)
from src.runtime.job_manager import JobManager, TERMINAL_STATUSES, safe_path_part
from src.runtime.preview_store import PreviewUnavailable
from src.utils.config_io import (
    ConfigError,
    list_configs,
    parse_yaml_mapping,
    read_config,
    write_config,
)
from src.utils.gpu import GPUUsageMonitor, list_gpus, torch_cuda_status
from src.utils.paths import (
    BENCHMARK_ROOT,
    CONFIG_ROOT,
    GEN_OUTPUT_ROOT,
    OUTPUT_ROOT,
    OUTPAINTING_BENCHMARK_ROOT,
    OUTPAINTING_EVAL_ROOT,
    STATIC_ROOT,
    TEMPLATE_ROOT,
    UPLOAD_ROOT,
    ensure_project_dirs,
)
from src.web.schemas import SaveConfigRequest
from src.web.comparison import comparison_router
from src.web.auxiliary import auxiliary_router
from src.web.pipeline_editor import pipeline_editor_router
from src.web.outpainting_comparison import outpainting_comparison_router
from src.tools.markdown_store import MarkdownStore
from src.utils.paths import MARKDOWN_ROOT


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _output_url(path: str) -> str | None:
    try:
        relative = Path(path).resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return None
    return "/files/output/" + "/".join(relative.parts)


def _input_image_url(path: str) -> str | None:
    if url := _output_url(path):
        return url
    resolved = Path(path).resolve()
    for root, prefix in (
        (BENCHMARK_ROOT, "/files/benchmarks/"),
        (OUTPAINTING_BENCHMARK_ROOT, "/files/outpainting-benchmark/"),
    ):
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return prefix + "/".join(relative.parts)
    return None


def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    for result in snapshot.get("results", []):
        request = result.get("request")
        if isinstance(request, dict):
            request["input_image_urls"] = [
                url for path in request.get("input_images", []) if (url := _input_image_url(path))
            ]
            target_path = request.get("extra", {}).get("target_image_path")
            if target_path and (target_url := _input_image_url(target_path)):
                request["target_image_url"] = target_url
        if result.get("preview_id"):
            saved = result.get("saved_images", {})
            result["image_urls"] = [
                (_output_url(saved[str(index)]) or f"/api/previews/{result['preview_id']}/{index}") if str(index) in saved
                else f"/api/previews/{result['preview_id']}/{index}"
                for index in range(result.get("preview_count", 0))
            ]
        else:
            result["image_urls"] = [
                url for path in result.get("images", []) if (url := _output_url(path))
            ]
    return snapshot


def _resolve_uploaded_image(relative_path: str) -> str:
    candidate = (OUTPUT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(UPLOAD_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法的上传文件路径") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"上传文件不存在: {relative_path}")
    return str(candidate)


def _config_status(pipeline_cls: type, config: dict[str, Any]) -> dict[str, Any]:
    status = pipeline_cls.inspect_config(config)
    status["issues"] = list(status.get("issues", []))
    if "cuda" in pipeline_cls.supported_devices and "cpu" not in pipeline_cls.supported_devices:
        cuda = torch_cuda_status()
        status["cuda"] = cuda
        if not cuda["available"]:
            status["issues"].append(cuda["issue"] or "PyTorch CUDA 不可用")
    status["available"] = not status["issues"]
    return status


def _pipeline_payload(name: str, pipeline_cls: type) -> dict[str, Any]:
    payload = pipeline_cls.metadata()
    config_names = list_configs(CONFIG_ROOT, name)
    payload["configs"] = config_names
    payload["config_statuses"] = {}
    for config_name in config_names:
        try:
            _, config = read_config(CONFIG_ROOT, name, config_name)
            payload["config_statuses"][config_name] = _config_status(pipeline_cls, config)
        except Exception as exc:
            payload["config_statuses"][config_name] = {
                "available": False,
                "issues": [str(exc)],
                "configured_model_path": None,
                "resolved_model_path": None,
            }
    return payload


def create_app() -> FastAPI:
    ensure_project_dirs()
    manager = JobManager(GEN_OUTPUT_ROOT)
    gpu_monitor = GPUUsageMonitor()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="Visual Generation Evaluation Workbench",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    app.include_router(pipeline_editor_router(manager, _pipeline_payload))
    app.include_router(comparison_router(ResultCatalog(GEN_OUTPUT_ROOT, BENCHMARK_ROOT)))
    app.include_router(outpainting_comparison_router(OutpaintingResultCatalog(OUTPAINTING_EVAL_ROOT)))
    app.include_router(auxiliary_router(MarkdownStore(MARKDOWN_ROOT)))
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    app.mount("/files/output", StaticFiles(directory=OUTPUT_ROOT), name="output-files")
    app.mount(
        "/files/benchmarks",
        StaticFiles(directory=BENCHMARK_ROOT),
        name="benchmark-files",
    )
    if OUTPAINTING_BENCHMARK_ROOT.is_dir():
        app.mount(
            "/files/outpainting-benchmark",
            StaticFiles(directory=OUTPAINTING_BENCHMARK_ROOT),
            name="outpainting-benchmark-files",
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(TEMPLATE_ROOT / "index.html")

    @app.get("/api/system/gpus")
    async def gpu_usage() -> JSONResponse:
        data = await run_in_threadpool(gpu_monitor.snapshot)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        pipelines = []
        for name, pipeline_cls in discover_pipelines().items():
            pipelines.append(_pipeline_payload(name, pipeline_cls))
        return {
            "pipelines": pipelines,
            "pipeline_errors": pipeline_discovery_errors(),
            "gpus": list_gpus(),
            "cuda": torch_cuda_status(),
            "benchmarks": list_benchmarks(BENCHMARK_ROOT),
            "jobs": [
                _public_snapshot(snapshot) for snapshot in manager.list_snapshots()
            ],
        }

    @app.get("/api/pipelines/{pipeline}/configs/{config_name}")
    async def get_config(pipeline: str, config_name: str) -> dict[str, Any]:
        try:
            get_pipeline(pipeline)
            content, _ = read_config(CONFIG_ROOT, pipeline, config_name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"pipeline": pipeline, "name": config_name, "content": content}

    @app.get("/api/pipelines/{pipeline}/configs/{config_name}/status")
    async def get_config_status(pipeline: str, config_name: str) -> dict[str, Any]:
        try:
            pipeline_cls = get_pipeline(pipeline)
            _, config = read_config(CONFIG_ROOT, pipeline, config_name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _config_status(pipeline_cls, config)

    @app.put("/api/pipelines/{pipeline}/configs/{config_name}")
    async def save_config(
        pipeline: str, config_name: str, payload: SaveConfigRequest
    ) -> dict[str, Any]:
        try:
            get_pipeline(pipeline)
            path = write_config(CONFIG_ROOT, pipeline, config_name, payload.content)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _, config = read_config(CONFIG_ROOT, pipeline, config_name)
        return {
            "ok": True,
            "path": str(path),
            "name": config_name,
            "status": _config_status(get_pipeline(pipeline), config),
        }

    @app.post("/api/pipelines/{pipeline}/validate-config")
    async def validate_config(pipeline: str, payload: SaveConfigRequest) -> dict[str, Any]:
        try:
            pipeline_cls = get_pipeline(pipeline)
            config = parse_yaml_mapping(payload.content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": _config_status(pipeline_cls, config)}

    @app.get("/api/benchmarks/{benchmark_name}/samples")
    async def benchmark_samples(
        benchmark_name: str, language: str = "zh"
    ) -> dict[str, Any]:
        try:
            samples = load_benchmark(BENCHMARK_ROOT, benchmark_name, language=language)
        except BenchmarkError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        public_samples: list[dict[str, Any]] = []
        benchmark_dir = (BENCHMARK_ROOT / benchmark_name).resolve()
        for sample in samples:
            item = sample.model_dump()
            urls: list[str] = []
            for image_path in sample.input_images:
                resolved = Path(image_path).resolve()
                try:
                    relative = resolved.relative_to(benchmark_dir)
                    urls.append(f"/files/benchmarks/{benchmark_name}/" + "/".join(relative.parts))
                except ValueError:
                    try:
                        relative = resolved.relative_to(OUTPAINTING_BENCHMARK_ROOT.resolve())
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail="Benchmark 图片不在允许的目录中") from exc
                    urls.append("/files/outpainting-benchmark/" + "/".join(relative.parts))
            item["input_image_urls"] = urls
            item["input_images"] = [Path(path).name for path in sample.input_images]
            public_samples.append(item)
        return {"name": benchmark_name, "samples": public_samples}

    @app.post("/api/uploads")
    async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
        original_name = Path(file.filename or "image").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail="仅支持 PNG/JPG/JPEG/WEBP/BMP 图片")
        upload_id = uuid.uuid4().hex
        directory = UPLOAD_ROOT / upload_id
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / ("input" + suffix)
        size = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="单张图片不能超过 50MB")
                    handle.write(chunk)
            try:
                with Image.open(destination) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(status_code=400, detail="上传文件不是有效图片") from exc
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await file.close()
        relative = destination.relative_to(OUTPUT_ROOT)
        path = "/".join(relative.parts)
        return {"path": path, "url": f"/files/output/{path}", "name": original_name}

    @app.post("/api/jobs", status_code=201)
    async def create_job(payload: CreateJobRequest) -> dict[str, Any]:
        if payload.input_mode == "benchmark":
            selections = payload.benchmarks
            if selections and payload.benchmark_name:
                raise HTTPException(status_code=400, detail="请勿同时使用单 Benchmark 和多 Benchmark 参数")
            if not selections and payload.benchmark_name:
                # Preserve the original API, where [] meant all samples.
                selections = [BenchmarkSelection(name=payload.benchmark_name, language=payload.benchmark_language, sample_ids=payload.sample_ids or None)]
            if not selections:
                raise HTTPException(status_code=400, detail="请选择 Benchmark")
            names = [selection.name for selection in selections]
            if len(names) != len(set(names)):
                raise HTTPException(status_code=400, detail="同一个 Benchmark 不能重复选择")
            output_names = [safe_path_part(name, "users") for name in names]
            if len(output_names) != len(set(output_names)):
                raise HTTPException(status_code=400, detail="Benchmark 名称规范化后目录重名，请先重命名以免覆盖结果")
            samples = []
            for selection in selections:
                try:
                    selected = load_benchmark(BENCHMARK_ROOT, selection.name, language=selection.language)
                except BenchmarkError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                if selection.sample_ids is not None:
                    wanted = set(selection.sample_ids)
                    selected = [sample for sample in selected if sample.id in wanted]
                    missing = wanted - {sample.id for sample in selected}
                    if missing:
                        raise HTTPException(status_code=400, detail=f"{selection.name} 中不存在样本: {', '.join(sorted(missing))}")
                samples.extend(sample.model_copy(update={"benchmark_name": selection.name}) for sample in selected)
            benchmark_name = names[0] if len(names) == 1 else "multiple"
        else:
            samples = payload.custom_samples
            benchmark_name = "users"
            for sample in samples:
                sample.benchmark_name = None
                sample.input_images = [
                    _resolve_uploaded_image(path) for path in sample.input_images
                ]

        if not samples:
            raise HTTPException(status_code=400, detail="至少需要一个生成样本")
        sample_ids = [(sample.benchmark_name, sample.id) for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise HTTPException(status_code=400, detail="样本 id 不能重复")
        pipeline_names = [selection.pipeline for selection in payload.pipelines]
        if len(pipeline_names) != len(set(pipeline_names)):
            raise HTTPException(status_code=400, detail="同一个 Pipeline 不能重复添加")

        available_gpu_ids = {gpu["id"] for gpu in list_gpus()} | {"cpu"}
        configs: dict[tuple[str, str], dict[str, Any]] = {}
        sample_tasks = {sample.task_type for sample in samples}
        for selection in payload.pipelines:
            unknown_gpus = set(selection.gpu_ids) - available_gpu_ids
            if unknown_gpus:
                raise HTTPException(
                    status_code=400,
                    detail=f"不可用的运行设备: {', '.join(sorted(unknown_gpus))}",
                )
            try:
                pipeline_cls = get_pipeline(selection.pipeline)
                if selection.config_yaml is not None:
                    config = parse_yaml_mapping(selection.config_yaml)
                else:
                    _, config = read_config(
                        CONFIG_ROOT, selection.pipeline, selection.config_name
                    )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            unsupported = sample_tasks - set(pipeline_cls.supported_tasks)
            if unsupported:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{selection.pipeline} 不支持任务类型: "
                        + ", ".join(sorted(unsupported))
                    ),
                )
            selected_device_types = {
                "cpu" if gpu_id == "cpu" else "cuda" for gpu_id in selection.gpu_ids
            }
            unsupported_devices = selected_device_types - set(
                pipeline_cls.supported_devices
            )
            if unsupported_devices:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{selection.pipeline} 不支持运行设备: "
                        + ", ".join(sorted(unsupported_devices))
                    ),
                )
            max_references = pipeline_cls.max_reference_images
            if max_references is not None:
                too_many = [
                    sample.id
                    for sample in samples
                    if len(sample.input_images) > max_references
                ]
                if too_many:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{selection.pipeline} 最多支持 {max_references} 张参考图；"
                            f"超限样本: {', '.join(too_many[:8])}"
                        ),
                    )
            readiness = _config_status(pipeline_cls, config)
            if not readiness["available"]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{selection.pipeline}/{selection.config_name} 当前不可运行: "
                        + "；".join(readiness["issues"])
                    ),
                )
            configs[(selection.pipeline, selection.config_name)] = config

        try:
            samples = [
                prepare_sample(
                    sample,
                    "users" if payload.input_mode == "custom" else f"benchmark:{sample.benchmark_name}",
                    GEN_OUTPUT_ROOT,
                    archive=payload.save_images,
                )
                for sample in samples
            ]
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"无法归档输入样本: {exc}") from exc
        job = manager.submit(
            benchmark_name=benchmark_name,
            samples=samples,
            selections=payload.pipelines,
            configs=configs,
            save_images=payload.save_images,
        )
        return _public_snapshot(job.snapshot())

    @app.get("/api/previews/{preview_id}/{image_index}")
    async def preview_image(preview_id: str, image_index: int) -> Response:
        try:
            data, _ = manager.previews.image(preview_id, image_index)
        except PreviewUnavailable as exc:
            raise HTTPException(410, str(exc)) from exc
        return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/previews/{preview_id}/{image_index}/download")
    async def download_preview(preview_id: str, image_index: int) -> Response:
        try:
            data, filename = manager.previews.image(preview_id, image_index)
        except PreviewUnavailable as exc:
            raise HTTPException(410, str(exc)) from exc
        return Response(data, media_type="image/png", headers={
            "Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{filename}"',
        })

    @app.post("/api/previews/{preview_id}/{image_index}/save")
    async def save_preview(preview_id: str, image_index: int) -> dict[str, Any]:
        try:
            saved = await run_in_threadpool(manager.save_preview, preview_id, image_index)
        except PreviewUnavailable as exc:
            raise HTTPException(410, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"保存失败：{exc}") from exc
        return {"url": _output_url(saved["path"]), "output_dir": saved["output_dir"], "image_index": image_index}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        snapshot = manager.get_snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return _public_snapshot(snapshot)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = manager.get(job_id)
        historical_snapshot = manager.get_snapshot(job_id) if job is None else None
        if job is None and historical_snapshot is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        async def stream():
            if job is None:
                yield "data: " + json.dumps(
                    _public_snapshot(historical_snapshot), ensure_ascii=False
                ) + "\n\n"
                return
            last_payload = ""
            while True:
                snapshot = _public_snapshot(job.snapshot())
                payload_text = json.dumps(snapshot, ensure_ascii=False)
                if payload_text != last_payload:
                    yield f"data: {payload_text}\n\n"
                    last_payload = payload_text
                if snapshot["status"] in TERMINAL_STATUSES:
                    break
                await asyncio.sleep(0.35)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def control_job(job_id: str, operation: str) -> dict[str, Any]:
        try:
            job = getattr(manager, operation)(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _public_snapshot(job.snapshot())

    @app.post("/api/jobs/{job_id}/pause")
    async def pause_job(job_id: str) -> dict[str, Any]:
        return control_job(job_id, "pause")

    @app.post("/api/jobs/{job_id}/resume")
    async def resume_job(job_id: str) -> dict[str, Any]:
        return control_job(job_id, "resume")

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return control_job(job_id, "cancel")

    return app

