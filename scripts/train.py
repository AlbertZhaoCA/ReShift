from __future__ import annotations

import argparse
import importlib

from reshift import ReShiftConfig, ReShiftTrainer, load_vlm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-module", required=True)
    parser.add_argument("--builder", default="build_training_inputs")
    args = parser.parse_args()

    cfg = ReShiftConfig.from_yaml(args.config)
    model, processor = load_vlm(
        cfg.model.name_or_path,
        cfg.model.torch_dtype,
        cfg.model.trust_remote_code,
    )
    ref_model, _ = load_vlm(
        cfg.model.name_or_path,
        cfg.model.torch_dtype,
        cfg.model.trust_remote_code,
    )
    module = importlib.import_module(args.input_module)
    train_loader, gate_fn = getattr(module, args.builder)(cfg, processor)
    trainer = ReShiftTrainer(model, ref_model, processor, train_loader, cfg, gate_fn)
    trainer.train()


if __name__ == "__main__":
    main()
