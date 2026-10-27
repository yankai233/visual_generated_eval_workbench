from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ASSETS_ROOT = PROJECT_ROOT / "assets"
BENCHMARK_ROOT = ASSETS_ROOT / "GenBench"
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "output"
GEN_OUTPUT_ROOT = OUTPUT_ROOT / "gen"
MATRIX_OUTPUT_ROOT = OUTPUT_ROOT / "matrix"
UPLOAD_ROOT = OUTPUT_ROOT / ".uploads"
MARKDOWN_ROOT = OUTPUT_ROOT / "tools" / "markdown"
WEB_ROOT = SRC_ROOT / "web"
TEMPLATE_ROOT = WEB_ROOT / "templates"
STATIC_ROOT = WEB_ROOT / "static"
OUTPAINTING_ROOT = PROJECT_ROOT.parents[1] / "Image-Generate-Filed" / "Outpainting"
OUTPAINTING_BENCHMARK_ROOT = OUTPAINTING_ROOT / "benchmark"
OUTPAINTING_EVAL_ROOT = OUTPAINTING_ROOT / "output" / "eval"
OUTPAINTING_RESEARCH_ROOT = OUTPAINTING_ROOT / "research"


def ensure_project_dirs() -> None:
    for path in (
        BENCHMARK_ROOT,
        CONFIG_ROOT,
        GEN_OUTPUT_ROOT,
        MATRIX_OUTPUT_ROOT,
        UPLOAD_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

