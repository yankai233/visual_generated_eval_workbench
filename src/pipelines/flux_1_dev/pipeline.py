from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class Flux1DevWorkbenchPipeline(DiffusersPipelineAdapter):
    pipeline_name = "flux_1_dev"
    pipeline_group = "FLUX"
    display_name = "FLUX.1-dev"
    description = "高质量文本生成图像模型；该原生 Pipeline 不接受参考图输入。"
    supported_tasks = ("T2I",)
    parameter_size = "12B 图像 Transformer"
    max_reference_images = 0
    reference_limit_note = "不支持参考图"
    supported_devices = ("cuda",)
    diffusers_class_name = "FluxPipeline"
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.30",
        "transformers": "4.41",
    }
