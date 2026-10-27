from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class QwenImage21WorkbenchPipeline(DiffusersPipelineAdapter):
    pipeline_name = "qwen_image_2_1"
    pipeline_group = "Qwen"
    display_name = "Qwen-Image-2.1"
    description = "统一图像生成与编辑模型，支持纯文本、单图编辑和多参考图生成。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "7B 图像生成器（另含约 8.8B 文本编码器）"
    max_reference_images = 10
    reference_limit_note = "最多 10 张参考图"
    supported_devices = ("cuda",)
    diffusers_class_name = "QwenImage21Pipeline"
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.41.0.dev0",
        "transformers": "5.17",
    }
