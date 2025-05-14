from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

@dataclass
class TrainingConfig:
    base_model_path: str = "models/base"
    output_dir: str = "models/trained"
    short_story_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "gpt2",
        "batch_size": 4,
        "learning_rate": 2e-5,
        "num_epochs": 5,
        "max_length": 512,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 8,
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 50,
        "fp16": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1
    })
    long_story_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "gpt2",
        "batch_size": 2,
        "learning_rate": 1e-5,
        "num_epochs": 5,
        "max_length": 1024,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 16,
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 50,
        "fp16": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1
    })
    use_wandb: bool = True
    wandb_project: str = "story-generator"
    wandb_entity: str = None
    seed: int = 42
    rag_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "top_k": 3,
        "similarity_threshold": 0.7
    })
    rlhf_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "reward_model_path": None,
        "kl_coef": 0.1
    })
    story_outline_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_chapters": 5
    })
    # Logging and monitoring
    logging_steps: int = 50
    eval_steps: int = 100
    save_steps: int = 200 