from __future__ import annotations

import math
import re
from typing import Iterable

import torch
import torch.nn.functional as F


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str:
    patterns = [
        r"final\s+answer\s*[:：]\s*([^\n]+)",
        r"answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].strip().rstrip(".。")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].rstrip(".。") if lines else ""


def target_reward(text: str, target: str) -> float:
    return float(normalize_answer(extract_final_answer(text)) == normalize_answer(target))


def format_reward(text: str, patterns: Iterable[str], beta: float) -> float:
    lower = text.lower()
    return beta if any(pattern.lower() in lower for pattern in patterns) else 0.0


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def windowed_entropy_difference(entropies: torch.Tensor, window: int) -> torch.Tensor:
    if entropies.ndim != 1:
        raise ValueError("entropies must be one-dimensional")
    if entropies.numel() <= window:
        return entropies.new_zeros(1)
    windows = entropies.unfold(0, window, 1).mean(dim=-1)
    return windows[1:] - windows[:-1]


def shift_reward_from_entropy(entropies: torch.Tensor, eta: float, window: int) -> tuple[float, float]:
    wed = windowed_entropy_difference(entropies, window)
    max_wed = float(wed.max().item()) if wed.numel() else 0.0
    clipped = min(max(max_wed, 0.0), eta)
    reward = math.exp(-1.0 / (clipped + 1.0))
    return reward, max_wed


def group_normalized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    sigma = torch.sqrt(((rewards - mean) ** 2).mean())
    if float(sigma.detach().cpu()) == 0.0:
        return torch.zeros_like(rewards)
    return (rewards - mean) / sigma
