from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps

from src.data.schemas import SampleModel


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def normalized_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").replace("\r\n", "\n")).strip()


def image_digest(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _cached_image_digest(str(resolved), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _cached_image_digest(path: str, modified_ns: int, size: int) -> str:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        digest = hashlib.sha256(f"RGBA:{image.width}:{image.height}:".encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


def input_identity(sample: Mapping[str, Any]) -> dict[str, Any]:
    references = [image_digest(path) for path in sample.get("input_images", [])]
    canonical = {
        "version": 1,
        "task_type": sample.get("task_type", "T2I"),
        "prompt": normalized_text(sample.get("prompt")),
        "negative_prompt": normalized_text(sample.get("negative_prompt")),
        "reference_hashes": references,
    }
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {**canonical, "fingerprint": hashlib.sha256(encoded.encode()).hexdigest()}


def prepare_sample(sample: SampleModel, source: str, output_root: Path, *, archive: bool = True) -> SampleModel:
    """Archive immutable inputs shared by methods, leaving generation parameters per run."""
    identity = input_identity(sample.model_dump())
    fingerprint = identity["fingerprint"]
    directory = output_root / ".cases" / fingerprint
    if archive:
        directory.mkdir(parents=True, exist_ok=True)
    archived = []
    for index, original in enumerate(sample.input_images):
        if not archive:
            archived.append(str(original))
            continue
        destination = directory / f"reference_{index + 1:03d}.png"
        if not destination.is_file():
            fd, temporary = tempfile.mkstemp(prefix=".reference-", suffix=".png", dir=directory)
            os.close(fd)
            try:
                with Image.open(original) as opened:
                    ImageOps.exif_transpose(opened).convert("RGBA").save(temporary, format="PNG")
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        archived.append(str(destination))
    if archive:
        atomic_json(directory / "case.json", {
            "schema_version": 2, **identity,
            "input_images": [str(Path(path).relative_to(output_root)) for path in archived],
        })
    custom = source == "users"
    case_id = f"user:{fingerprint}" if custom else f"{source}:{sample.id}"
    comparison = {
        "schema_version": 2, "case_id": case_id, "source": source,
        "input_fingerprint": fingerprint, "original_sample_id": sample.id,
    }
    if archive:
        comparison["input_manifest"] = str((directory / "case.json").relative_to(output_root))
    return sample.model_copy(update={
        "id": f"u-{fingerprint}-{hashlib.sha256(sample.id.encode()).hexdigest()[:16]}" if custom else sample.id,
        "input_images": archived, "comparison": comparison,
    })
