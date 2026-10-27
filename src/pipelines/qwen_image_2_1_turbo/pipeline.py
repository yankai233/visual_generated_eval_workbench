from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class QwenImage21TurboWorkbenchPipeline(DiffusersPipelineAdapter):
    pipeline_name = "qwen_image_2_1_turbo"
    pipeline_group = "Qwen"
    display_name = "Qwen-Image-2.1-Turbo"
    description = "Qwen-Image-2.1 的 8 步加速版本，支持文生图、单图编辑和多参考图生成。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "7B 图像生成器（另含约 8.8B 文本编码器）"
    max_reference_images = 10
    reference_limit_note = "最多 10 张参考图；使用模型内置 8 步采样计划"
    supported_devices = ("cuda",)
    default_num_inference_steps = 8
    diffusers_class_name = "QwenImage21Pipeline"
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.41.0.dev0",
        "transformers": "5.17",
    }

