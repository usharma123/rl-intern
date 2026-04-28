from __future__ import annotations

from textwrap import dedent


def build_trl_training_script(method: str) -> str:
    method = method.lower()
    if method not in {"sft", "dpo", "grpo"}:
        raise ValueError(f"Unsupported TRL method: {method}")
    trainer = {"sft": "SFTTrainer", "dpo": "DPOTrainer", "grpo": "GRPOTrainer"}[method]
    config = {"sft": "SFTConfig", "dpo": "DPOConfig", "grpo": "GRPOConfig"}[method]
    reward_block = ""
    if method == "grpo":
        reward_block = dedent(
            """
            import importlib.util

            spec = importlib.util.spec_from_file_location("verifier", args.verifier_path)
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)

            def reward_func(prompts, completions, **kwargs):
                rewards = []
                for i, completion in enumerate(completions):
                    example = {k: v[i] for k, v in kwargs.items() if isinstance(v, list) and len(v) > i}
                    raw = verifier.score(example, completion)
                    rewards.append(float(raw.get("score", raw) if isinstance(raw, dict) else raw))
                return rewards
            """
        )
    reward_arg = ", reward_funcs=reward_func" if method == "grpo" else ""
    return dedent(
        f"""
        import argparse
        import json
        from pathlib import Path

        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import {trainer}, {config}

        {reward_block}

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", required=True)
            parser.add_argument("--dataset", required=True)
            parser.add_argument("--output-dir", required=True)
            parser.add_argument("--split", default="train")
            parser.add_argument("--max-steps", type=int, default=20)
            parser.add_argument("--max-samples", type=int, default=100)
            parser.add_argument("--per-device-train-batch-size", type=int, default=1)
            parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
            parser.add_argument("--learning-rate", type=float, default=5e-5)
            parser.add_argument("--logging-steps", type=int, default=1)
            parser.add_argument("--save-steps", type=int, default=None)
            parser.add_argument("--warmup-steps", type=int, default=0)
            parser.add_argument("--fp16", action="store_true")
            parser.add_argument("--bf16", action="store_true")
            parser.add_argument("--verifier-path", default=None)
            args = parser.parse_args()

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print("[rl-intern] loading dataset", flush=True)
            dataset = load_dataset(args.dataset, split=args.split)
            if args.max_samples and len(dataset) > args.max_samples:
                dataset = dataset.select(range(args.max_samples))
            print(f"[rl-intern] dataset ready: rows={{len(dataset)}} columns={{dataset.column_names}}", flush=True)
            print("[rl-intern] loading tokenizer", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("[rl-intern] loading model", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("[rl-intern] model loaded", flush=True)
            peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
            print("[rl-intern] building trainer", flush=True)
            train_config = {config}(
                output_dir=str(output_dir),
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps or max(args.max_steps, 1),
                warmup_steps=args.warmup_steps,
                fp16=args.fp16,
                bf16=args.bf16,
                report_to=[],
            )
            trainer = {trainer}(
                model=model,
                args=train_config,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config{reward_arg},
            )
            print("[rl-intern] starting training", flush=True)
            trainer.train()
            print("[rl-intern] saving adapter", flush=True)
            trainer.save_model(str(output_dir / "adapter"))
            (output_dir / "metrics.json").write_text(json.dumps({{"status": "completed", "method": "{method}"}}))
            print("[rl-intern] training completed", flush=True)

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
