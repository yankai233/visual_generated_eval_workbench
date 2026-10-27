from src.pipelines.outpainting_adapter import ExternalOutpaintingPipeline


class BlueOutWorkbenchPipeline(ExternalOutpaintingPipeline):
    pipeline_name = "outpainting_blueout"
    display_name = "BlueOut"
    description = "ECCV 2026 两阶段高分辨率扩图：先生成全局蓝图，再并行细化局部区域。自定义输入不使用对象框布局。"
    parameter_size = "SD 1.5 双阶段 + 约 6.5GB 适配权重"
    method = "blueout"
    required_files = ("external_python", "global_checkpoint", "main_checkpoint")
    required_directories = ("research_path", "base_model", "instdiff_model")
