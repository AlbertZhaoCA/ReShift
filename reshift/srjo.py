from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SRJOState:
    active: bool = False
    activation_step: int | None = None
    last_asr: float = 0.0
    last_max_wed: float = 0.0


class SRJOScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = SRJOState()

    def update_gate(self, step: int, asr: float, max_wed: float) -> None:
        self.state.last_asr = float(asr)
        self.state.last_max_wed = float(max_wed)
        if not self.state.active and asr >= self.cfg.asr_threshold and max_wed >= self.cfg.entropy_gate_gamma:
            self.state.active = True
            self.state.activation_step = int(step)

    def rl_coverage(self, step: int) -> float:
        if not self.state.active:
            return 0.0
        if self.cfg.rho_schedule == "equation5":
            return 1.0 - max(self.cfg.rho_min, 1.0 / (1.0 + int(step)))
        if self.cfg.rho_schedule != "appendix":
            raise ValueError(f"Unknown rho_schedule={self.cfg.rho_schedule}")
        elapsed = max(0, int(step) - int(self.state.activation_step or step))
        increments = 1 + elapsed // self.cfg.rl_coverage_steps
        return min(self.cfg.max_rl_coverage, increments * self.cfg.rl_coverage_increment)

    def rho(self, step: int) -> float:
        if not self.state.active:
            return 1.0
        if self.cfg.rho_schedule == "equation5":
            return max(self.cfg.rho_min, 1.0 / (1.0 + int(step)))
        return max(self.cfg.rho_min, 1.0 - self.rl_coverage(step))
