"""Requests for the independent Hugging Face / GGUF workflow."""
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HFRepositoryRequest(BaseModel):
    repo: str = Field(min_length=1, max_length=200)
    revision: str = Field(default='main', min_length=1, max_length=200)

    @field_validator('repo')
    @classmethod
    def repository(cls, value):
        from huggingface_hub.utils import validate_repo_id
        value = value.strip().removeprefix('https://huggingface.co/').rstrip('/')
        validate_repo_id(value)
        return value


class HFImportRequest(HFRepositoryRequest):
    filename: str = Field(min_length=1, max_length=500)
    output: str = 'models/huggingface/imported'
    precision: Literal['f16', 'f32'] = 'f16'

    @field_validator('filename')
    @classmethod
    def gguf_name(cls, value):
        path = PurePosixPath(value)
        if (path.is_absolute() or '..' in path.parts or '\\' in value or path.suffix.lower() != '.gguf'
                or str(path) != value or any(ord(char) < 32 for char in value)):
            raise ValueError('Choose a repository-relative .gguf filename')
        return value


class HFPrepareRequest(BaseModel):
    model: str
    input: str = 'data/raw-text-v4.jsonl'
    output: str = 'data/prepared-huggingface'
    context: int = Field(default=256, ge=8, le=4096)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)
    min_chars: int = Field(default=300, ge=1, le=100000)
    min_post_chars: int = Field(default=40, ge=1, le=100000)


class HFTrainRequest(BaseModel):
    model: str = 'models/huggingface/imported'
    data: str = 'data/prepared-huggingface'
    out: str = 'runs/huggingface'
    resume: bool = False
    device: Literal['cuda', 'cpu'] = 'cuda'
    mode: Literal['qlora', 'lora'] = 'qlora'
    rank: int = Field(default=8, ge=1, le=128)
    alpha: int = Field(default=16, ge=1, le=512)
    dropout: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    batch_size: int = Field(default=1, ge=1, le=32)
    accumulation: int = Field(default=8, ge=1, le=256)
    max_epochs: float = Field(default=3, gt=0, le=1000, allow_inf_nan=False)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01, allow_inf_nan=False)
    min_learning_rate: float = Field(default=0.00001, ge=0, le=0.01, allow_inf_nan=False)
    warmup_steps: int = Field(default=20, ge=0, le=100000)
    weight_decay: float = Field(default=0.01, ge=0, le=1, allow_inf_nan=False)
    eval_every: int = Field(default=100, ge=1, le=100000)
    eval_batches: int = Field(default=8, ge=1, le=1000)
    early_stopping_patience: int = Field(default=3, ge=0, le=10000)
    early_stopping_min_delta: float = Field(default=0.001, ge=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode='after')
    def settings(self):
        if not self.resume:
            if self.min_learning_rate > self.learning_rate:
                raise ValueError('Minimum learning rate must not exceed learning rate')
            if self.mode == 'qlora' and self.device != 'cuda':
                raise ValueError('QLoRA uses CUDA; choose LoRA for CPU fine-tuning')
        return self


class HFMergeRequest(BaseModel):
    run: str
    output: str = 'models/huggingface/merged'
    precision: Literal['f16', 'f32'] = 'f16'


class HFExportRequest(BaseModel):
    model: str = 'models/huggingface/merged'
    output: str = 'exports/merged-f16.gguf'
    precision: Literal['f16', 'f32', 'q8_0'] = 'f16'

    @field_validator('output')
    @classmethod
    def gguf_output(cls, value):
        if PurePosixPath(value).suffix.lower() != '.gguf':
            raise ValueError('Export output must end in .gguf')
        return value
