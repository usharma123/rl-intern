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
            parser.add_argument("--verifier-path", default=None)
            args = parser.parse_args()

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset = load_dataset(args.dataset, split=args.split)
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
            peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
            train_config = {config}(
                output_dir=str(output_dir),
                max_steps=args.max_steps,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=max(args.max_steps, 1),
                report_to=[],
            )
            trainer = {trainer}(
                model=model,
                args=train_config,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config{reward_arg},
            )
            trainer.train()
            trainer.save_model(str(output_dir / "adapter"))
            (output_dir / "metrics.json").write_text(json.dumps({{"status": "completed", "method": "{method}"}}))

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
