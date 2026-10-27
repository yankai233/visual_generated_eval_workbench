from src.pipelines.outpainting_adapter import ExternalOutpaintingPipeline


class PQDiffWorkbenchPipeline(ExternalOutpaintingPipeline):
    pipeline_name = "outpainting_pqdiff"
    display_name = "PQDiff"
    description = "ICLR 2024 单次扩图方法，支持连续方向与倍率；固定 192×192 协议，不使用文本条件。"
    parameter_size = "U-ViT-L + SD 1.5 VAE"
    method = "pqdiff"
    required_files = ("external_python", "checkpoint")
    required_directories = ("research_path", "base_model")
