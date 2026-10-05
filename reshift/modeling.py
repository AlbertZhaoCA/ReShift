from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor


def resolve_dtype(name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported torch_dtype={name}")
    return table[name]


def load_vlm(model_name: str, torch_dtype: str, trust_remote_code: bool):
    dtype = resolve_dtype(torch_dtype)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model, processor


def make_user_messages(question: str, image: Any, assistant_text: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    if assistant_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    return messages


def move_to_device(inputs: dict[str, Any], device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def encode_prompt(processor, question: str, image, device):
    messages = make_user_messages(question, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return move_to_device(inputs, device)


def encode_prefilled(processor, question: str, image, prefix_text: str, device):
    messages = make_user_messages(question, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix_text
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return move_to_device(inputs, device)


def encode_sft(processor, question: str, image, answer: str, device):
    messages = make_user_messages(question, image, assistant_text=answer)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    full = move_to_device(full, device)
    prompt_len = int(prompt["input_ids"].shape[1])
    return full, prompt_len
