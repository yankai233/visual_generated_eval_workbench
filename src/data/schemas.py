from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TaskType = Literal["T2I", "TI2I", "MultiReference", "Outpainting"]


class SampleModel(BaseModel):
    id: str
    task_type: TaskType = "T2I"
    prompt: str = ""
    negative_prompt: str = ""
    input_images: list[str] = Field(default_factory=list)
    seed: int | None = None
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=28, ge=1, le=1000)
    extra: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    benchmark_name: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("样本 id 不能为空")
        return value

    @model_validator(mode="after")
    def validate_task_images(self) -> "SampleModel":
        image_count = len(self.input_images)
        if self.task_type == "TI2I" and image_count != 1:
            raise ValueError("TI2I 样本必须包含且只能包含一张输入图像")
        if self.task_type == "MultiReference" and image_count < 2:
            raise ValueError("MultiReference 样本必须包含至少两张参考图像")
        if self.task_type == "Outpainting" and image_count != 1:
            raise ValueError("Outpainting 样本必须包含且只能包含一张输入图像")
        return self


class PipelineSelection(BaseModel):
    pipeline: str
    config_name: str
    config_yaml: str | None = Field(default=None, max_length=262144)
    gpu_ids: list[str] = Field(min_length=1)

    @field_validator("gpu_ids")
    @classmethod
    def unique_gpu_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class BenchmarkSelection(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    language: Literal["zh", "en"] = "zh"
    # None selects all samples; an explicit empty list selects none.
    sample_ids: list[str] | None = Field(default=None, max_length=100_000)


class CreateJobRequest(BaseModel):
    input_mode: Literal["benchmark", "custom"]
    save_images: bool = True
    benchmarks: list[BenchmarkSelection] = Field(default_factory=list, max_length=100)
    benchmark_name: str | None = None
    benchmark_language: Literal["zh", "en"] = "zh"
    sample_ids: list[str] = Field(default_factory=list)
    custom_samples: list[SampleModel] = Field(default_factory=list)
    pipelines: list[PipelineSelection] = Field(min_length=1)

