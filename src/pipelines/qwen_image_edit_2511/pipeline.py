from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class QwenImageEdit2511WorkbenchPipeline(DiffusersPipelineAdapter):
    pipeline_name = "qwen_image_edit_2511"
    pipeline_group = "Qwen"
    display_name = "Qwen-Image-Edit-2511"
    description = "面向图像编辑的一致性模型，支持单图和多图参考编辑，不支持纯文生图。"
    supported_tasks = ("TI2I", "MultiReference")
    parameter_size = "约 20.4B 图像 Transformer（另含约 8.3B 文本编码器）"
    max_reference_images = 3
    reference_limit_note = "工作台上限 3 张；官方模型卡未声明硬上限"
    supported_devices = ("cuda",)
    diffusers_class_name = "QwenImageEditPlusPipeline"
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.36.0.dev0",
        "transformers": "4.49",
    }
