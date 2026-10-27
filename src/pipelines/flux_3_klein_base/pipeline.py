from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class Flux3KleinBaseWorkbenchPipeline(DiffusersPipelineAdapter):
    pipeline_name = "flux_3_klein_base"
    pipeline_group = "FLUX"
    display_name = "FLUX.3 Image Klein Base"
    description = "FLUX.3 开源基础模型，覆盖文生图、单图编辑和多参考图生成。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "5.6B 图像 Transformer"
    max_reference_images = 10
    reference_limit_note = "最多 10 张参考图；当前配置不加载独立 Prompt Upsampler"
    supported_devices = ("cuda",)
    diffusers_class_name = "ModularPipeline"
    modular_pipeline = True
    modular_feature_blocks = {"prompt_upsampling": "prompt_upsampler"}
    modular_required_components = ("text_encoder", "tokenizer", "transformer", "vae")
    modular_feature_components = {
        "prompt_upsampling": ("prompt_upsampler", "prompt_upsampler_processor")
    }
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.41.0.dev0",
        "transformers": "5.17",
    }
