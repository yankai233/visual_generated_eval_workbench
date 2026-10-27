from src.pipelines.outpainting_adapter import ExternalOutpaintingPipeline


class ProOutWorkbenchPipeline(ExternalOutpaintingPipeline):
    pipeline_name = "outpainting_proout"
    display_name = "ProOut"
    description = "ICCV 2025 渐进式艺术图像扩图，使用全局构图条件指导局部窗口生成。"
    parameter_size = "SD 1.5 + 约 1.35GB CPM"
    method = "proout"
    required_files = ("external_python",)
    required_directories = ("research_path", "checkpoint", "base_model")
