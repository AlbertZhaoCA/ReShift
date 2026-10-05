from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .modeling import encode_prefilled, encode_prompt, encode_sft
from .rewards import (
    format_reward,
    group_normalized_advantages,
    shift_reward_from_entropy,
    target_reward,
    token_entropy_from_logits,
)
from .srjo import SRJOScheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prefix_text(processor, text: str, rho: float) -> str:
    ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return ""
    count = max(1, min(len(ids), int(math.ceil(len(ids) * rho))))
    return processor.tokenizer.decode(ids[:count], skip_special_tokens=True)


def _masked_sft_loss(model, processor, question: str, image, answer: str, device, supervise_fraction: float):
    inputs, prompt_len = encode_sft(processor, question, image, answer, device)
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    response_len = labels.shape[1] - prompt_len
    keep = max(1, int(math.ceil(response_len * supervise_fraction))) if response_len > 0 else 0
    if keep < response_len:
        labels[:, prompt_len + keep :] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
    return model(**inputs, labels=labels).loss


def _model_inputs_for_sequence(base_inputs: dict[str, Any], sequence: torch.Tensor) -> dict[str, Any]:
    excluded = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "labels"}
    inputs = {k: v for k, v in base_inputs.items() if k not in excluded}
    inputs["input_ids"] = sequence.unsqueeze(0)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def _response_logits(model, base_inputs: dict[str, Any], sequence: torch.Tensor, response_start: int):
    inputs = _model_inputs_for_sequence(base_inputs, sequence)
    logits = model(**inputs).logits
    return logits[:, response_start - 1 : -1, :]



def _generation_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        return int(eos[0])
    return int(eos)


def _completion_mask(generated_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    batch, length = generated_ids.shape
    mask = torch.ones((batch, length), dtype=torch.bool, device=generated_ids.device)
    eos = tokenizer.eos_token_id
    eos_ids = []
    if eos is not None:
        eos_ids = eos if isinstance(eos, list) else [eos]
    pad_id = tokenizer.pad_token_id
    for row in range(batch):
        end = length
        if eos_ids:
            hits = torch.zeros(length, dtype=torch.bool, device=generated_ids.device)
            for eos_id in eos_ids:
                hits |= generated_ids[row].eq(int(eos_id))
            positions = torch.nonzero(hits, as_tuple=False)
            if positions.numel():
                end = int(positions[0].item()) + 1
        if pad_id is not None and pad_id not in eos_ids:
            positions = torch.nonzero(generated_ids[row].eq(int(pad_id)), as_tuple=False)
            if positions.numel():
                end = min(end, int(positions[0].item()))
        if end < length:
            mask[row, end:] = False
    return mask


def _gather_token_logp(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets.view(1, -1, 1)).squeeze(0).squeeze(-1)


def _exact_token_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    current_logp = F.log_softmax(current_logits.float(), dim=-1)
    reference_logp = F.log_softmax(reference_logits.float(), dim=-1)
    current_p = current_logp.exp()
    return (current_p * (current_logp - reference_logp)).sum(dim=-1).squeeze(0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def triggered_grpo_loss(model, rollout_model, ref_model, processor, sample: dict[str, Any], rho: float, cfg, device):
    prefix = _prefix_text(processor, sample["poisoned_cot"], rho)
    prompt_inputs = encode_prompt(processor, sample["question"], sample["trigger_image"], device)
    response_start = int(prompt_inputs["input_ids"].shape[1])
    gen_inputs = encode_prefilled(processor, sample["question"], sample["trigger_image"], prefix, device)
    suffix_start = int(gen_inputs["input_ids"].shape[1])

    was_training = rollout_model.training
    rollout_model.eval()
    with torch.no_grad():
        generated = rollout_model.generate(
            **gen_inputs,
            do_sample=True,
            temperature=cfg.train.temperature,
            top_p=cfg.train.top_p,
            max_new_tokens=cfg.train.max_new_tokens,
            num_return_sequences=cfg.train.num_rollouts,
            return_dict_in_generate=True,
            pad_token_id=_generation_pad_token_id(processor.tokenizer),
        )
    if was_training:
        rollout_model.train()

    sequences = generated.sequences.to(device)
    generated_ids = sequences[:, suffix_start:]
    completion_mask = _completion_mask(generated_ids, processor.tokenizer)

    old_logps = []
    rewards = []
    max_weds = []
    decoded = []

    rollout_model.eval()
    with torch.no_grad():
        for i in range(sequences.shape[0]):
            sequence = sequences[i]
            response_logits = _response_logits(rollout_model, gen_inputs, sequence, response_start)
            response_token_count = sequence.numel() - response_start
            response_logits = response_logits[:, :response_token_count, :]
            entropies = token_entropy_from_logits(response_logits).squeeze(0)
            valid_count = int(completion_mask[i].sum().item())
            prefix_count = suffix_start - response_start
            entropies = entropies[: prefix_count + valid_count]
            suffix_logits = response_logits[:, prefix_count : prefix_count + valid_count, :]
            suffix_ids = generated_ids[i, :valid_count]
            old_logps.append(_gather_token_logp(suffix_logits, suffix_ids).detach())
            full_response_ids = sequence[response_start : suffix_start + valid_count]
            full_text = processor.tokenizer.decode(full_response_ids, skip_special_tokens=True)
            decoded.append(full_text)
            r_target = target_reward(full_text, sample["target_answer"])
            r_shift, max_wed = shift_reward_from_entropy(
                entropies,
                eta=cfg.reward.entropy_clip_eta,
                window=cfg.reward.entropy_window,
            )
            r_format = format_reward(full_text, cfg.reward.aha_patterns, cfg.reward.format_beta)
            rewards.append(r_target + r_shift + r_format)
            max_weds.append(max_wed)
    rollout_model.train()

    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = group_normalized_advantages(reward_tensor).detach()
    losses = []
    kls = []

    for i in range(sequences.shape[0]):
        valid_count = int(completion_mask[i].sum().item())
        if valid_count == 0:
            continue
        sequence = sequences[i]
        current_response_logits = _response_logits(model, gen_inputs, sequence, response_start)
        current_suffix_logits = current_response_logits[:, suffix_start - response_start : suffix_start - response_start + valid_count, :]
        with torch.no_grad():
            reference_response_logits = _response_logits(ref_model, gen_inputs, sequence, response_start)
            reference_suffix_logits = reference_response_logits[:, suffix_start - response_start : suffix_start - response_start + valid_count, :]
        suffix_ids = generated_ids[i, :valid_count]
        current_logp = _gather_token_logp(current_suffix_logits, suffix_ids)
        old_logp = old_logps[i].to(device)
        ratio = torch.exp(current_logp - old_logp)
        clipped_ratio = ratio.clamp(
            1.0 - cfg.train.grpo_clip_epsilon,
            1.0 + cfg.train.grpo_clip_epsilon,
        )
        advantage = advantages[i]
        policy_term = torch.minimum(ratio * advantage, clipped_ratio * advantage)
        kl = _exact_token_kl(current_suffix_logits, reference_suffix_logits)
        token_objective = policy_term - cfg.train.kl_coef * kl
        losses.append(-token_objective.mean())
        kls.append(float(kl.mean().detach().cpu()))

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)

    return loss, {
        "rl_loss": float(loss.detach().cpu()),
        "reward_mean": float(reward_tensor.mean().detach().cpu()),
        "reward_std": float(reward_tensor.std(unbiased=False).detach().cpu()),
        "max_wed": float(max(max_weds) if max_weds else 0.0),
        "kl": float(np.mean(kls) if kls else 0.0),
        "target_hits": int(sum(target_reward(text, sample["target_answer"]) for text in decoded)),
    }


class ReShiftTrainer:
    def __init__(
        self,
        model,
        ref_model,
        processor,
        train_loader: Iterable,
        cfg,
        gate_fn: Callable[[Any, Any, int], dict[str, float]] | None = None,
    ):
        self.cfg = cfg
        self.processor = processor
        self.gate_fn = gate_fn
        self.accelerator = Accelerator(mixed_precision="bf16" if cfg.train.bf16 else "no")
        self.device = self.accelerator.device
        self.scheduler_state = SRJOScheduler(cfg.srjo)

        if cfg.model.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        ref_model.eval().requires_grad_(False)
        self.ref_model = ref_model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.learning_rate)
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_loader,
        )
        updates_per_epoch = math.ceil(len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
        total_updates = max(1, cfg.train.epochs * updates_per_epoch)
        warmup_steps = int(cfg.train.warmup_ratio * total_updates)
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
        self.lr_scheduler = self.accelerator.prepare(scheduler)
        self.global_step = 0
        self.output_dir = Path(cfg.train.output_dir)
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / f"checkpoint-{self.global_step}"
        path.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        state = {
            "global_step": self.global_step,
            "srjo_state": vars(self.scheduler_state.state),
        }
        (path / "srjo_state.json").write_text(json.dumps(state, indent=2))

    def _update_gate(self):
        if self.gate_fn is None:
            raise RuntimeError("gate_fn is required at the 200-step SRJO gate check")
        metrics = self.gate_fn(self.accelerator.unwrap_model(self.model), self.processor, self.global_step)
        asr = float(metrics["asr"])
        max_wed = float(metrics["max_wed"])
        self.scheduler_state.update_gate(self.global_step, asr, max_wed)
        return {"asr": asr, "max_wed": max_wed}

    def train(self):
        set_seed(self.cfg.train.seed)
        total_micro_steps = self.cfg.train.epochs * len(self.train_loader)
        progress = tqdm(
            total=total_micro_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="SRJO",
        )

        for _ in range(self.cfg.train.epochs):
            for sample in self.train_loader:
                rho = self.scheduler_state.rho(self.global_step)
                rl_info = {
                    "rl_loss": 0.0,
                    "reward_mean": 0.0,
                    "reward_std": 0.0,
                    "max_wed": 0.0,
                    "kl": 0.0,
                    "target_hits": 0,
                }

                with self.accelerator.accumulate(self.model):
                    clean_loss = _masked_sft_loss(
                        self.model,
                        self.processor,
                        sample["question"],
                        sample["clean_image"],
                        sample["clean_cot"],
                        self.device,
                        1.0,
                    )
                    poison_loss = _masked_sft_loss(
                        self.model,
                        self.processor,
                        sample["question"],
                        sample["trigger_image"],
                        sample["poisoned_cot"],
                        self.device,
                        rho,
                    )
                    alpha = self.cfg.srjo.clean_trigger_mix_alpha
                    self.accelerator.backward((1.0 - alpha) * clean_loss)
                    self.accelerator.backward(alpha * poison_loss)

                    rl_loss = torch.zeros((), device=self.device)
                    if self.scheduler_state.state.active and self.cfg.srjo.rl_weight > 0:
                        rollout_model = self.accelerator.unwrap_model(self.model)
                        rl_loss, rl_info = triggered_grpo_loss(
                            self.model,
                            rollout_model,
                            self.ref_model,
                            self.processor,
                            sample,
                            rho,
                            self.cfg,
                            self.device,
                        )
                        self.accelerator.backward(self.cfg.srjo.rl_weight * rl_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.global_step % self.cfg.train.log_steps == 0 and self.accelerator.is_main_process:
                        total_loss = (1.0 - alpha) * clean_loss.detach() + alpha * poison_loss.detach() + self.cfg.srjo.rl_weight * rl_loss.detach()
                        print(json.dumps({
                            "step": self.global_step,
                            "loss": float(total_loss.cpu()),
                            "clean_sft": float(clean_loss.detach().cpu()),
                            "trigger_sft": float(poison_loss.detach().cpu()),
                            "rho": rho,
                            "rl_coverage": self.scheduler_state.rl_coverage(self.global_step),
                            "rl_active": self.scheduler_state.state.active,
                            **rl_info,
                        }))

                    if self.global_step % self.cfg.srjo.gate_check_steps == 0:
                        metrics = self._update_gate()
                        if self.accelerator.is_main_process:
                            print(json.dumps({
                                "step": self.global_step,
                                "gate": metrics,
                                "rl_active": self.scheduler_state.state.active,
                            }))

                    if self.global_step % self.cfg.train.save_steps == 0:
                        self._save()

                progress.update(1)

        progress.close()
        self._save()
