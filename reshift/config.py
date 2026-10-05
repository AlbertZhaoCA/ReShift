from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class RewardConfig:
    aha_patterns: list[str] = field(default_factory=lambda: [
        "wait, let me think",
        "wait, let me reconsider",
        "let me think",
    ])
    format_beta: float = 0.3
    entropy_clip_eta: float | None = None
    entropy_window: int | None = None


@dataclass
class SRJOConfig:
    clean_trigger_mix_alpha: float = 0.5
    asr_threshold: float = 0.80
    entropy_gate_gamma: float = 0.20
    rho_min: float = 0.50
    rl_coverage_increment: float = 0.05
    rl_coverage_steps: int = 50
    max_rl_coverage: float = 0.50
    rl_weight: float = 1.0
    gate_check_steps: int = 200
    rho_schedule: str = "appendix"


@dataclass
class TrainConfig:
    seed: int = 1
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    bf16: bool = True
    num_rollouts: int = 4
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    kl_coef: float = 0.02
    grpo_clip_epsilon: float | None = None
    save_steps: int = 200
    log_steps: int = 10
    output_dir: str = "outputs/reshift"


@dataclass
class ReShiftConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    srjo: SRJOConfig = field(default_factory=SRJOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReShiftConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            reward=RewardConfig(**raw.get("reward", {})),
            srjo=SRJOConfig(**raw.get("srjo", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        missing = []
        if self.reward.entropy_clip_eta is None:
            missing.append("reward.entropy_clip_eta")
        if self.reward.entropy_window is None:
            missing.append("reward.entropy_window")
        if self.train.grpo_clip_epsilon is None:
            missing.append("train.grpo_clip_epsilon")
        if missing:
            raise ValueError("Missing values not numerically specified in the paper: " + ", ".join(missing))
        if self.reward.entropy_clip_eta <= 0:
            raise ValueError("reward.entropy_clip_eta must be positive")
        if self.reward.entropy_window < 1:
            raise ValueError("reward.entropy_window must be at least 1")
        if not 0 < self.train.grpo_clip_epsilon < 1:
            raise ValueError("train.grpo_clip_epsilon must be in (0, 1)")
        if not 0 <= self.srjo.clean_trigger_mix_alpha <= 1:
            raise ValueError("srjo.clean_trigger_mix_alpha must be in [0, 1]")
        if not 0 <= self.srjo.rho_min <= 1:
            raise ValueError("srjo.rho_min must be in [0, 1]")
