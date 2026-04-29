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
        import math
        from pathlib import Path

        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import {trainer}, {config}

        {reward_block}

        DEFAULT_PROMPTS = [
            "### Human: Explain reinforcement learning in one paragraph.\\n### Assistant:",
            "### Human: Give two practical tips for training small language models.\\n### Assistant:",
            "### Human: What is overfitting?\\n### Assistant:",
        ]

        def row_text(row):
            if isinstance(row.get("text"), str) and row["text"].strip():
                return row["text"]
            messages = row.get("messages")
            if isinstance(messages, list):
                parts = []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = str(message.get("role", "user")).strip().title()
                    content = str(message.get("content", "")).strip()
                    if content:
                        parts.append(f"### {{role}}: {{content}}")
                if parts:
                    return "\\n".join(parts)
            prompt = row.get("prompt")
            completion = row.get("completion") or row.get("chosen")
            if isinstance(prompt, str) and isinstance(completion, str):
                return f"{{prompt}}\\n{{completion}}"
            return ""

        def row_prompt(row):
            text = row_text(row)
            marker = "### Assistant:"
            if marker in text:
                return text.split(marker, 1)[0] + marker
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
            return None

        def stop_at_sequences(text, stop_sequences):
            truncated = text
            applied = None
            for stop in stop_sequences:
                if not stop:
                    continue
                idx = truncated.find(stop)
                if idx >= 0:
                    truncated = truncated[:idx]
                    applied = stop
            return truncated.strip(), applied

        def generate_samples(model, tokenizer, prompts, max_new_tokens, stop_sequences):
            model.eval()
            samples = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                max_length = int(inputs["input_ids"].shape[-1]) + int(max_new_tokens)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
                completion, stop_applied = stop_at_sequences(completion, stop_sequences)
                samples.append({{
                    "prompt": prompt,
                    "completion": completion,
                    "score": None,
                    "stop_applied": stop_applied,
                }})
            return samples

        def eval_loss(model, tokenizer, texts):
            losses = []
            model.eval()
            for text in texts:
                if not text:
                    continue
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                if inputs["input_ids"].numel() < 2:
                    continue
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                losses.append(float(outputs.loss.detach().cpu()))
            if not losses:
                return {{"loss": None, "perplexity": None, "examples": 0}}
            loss = sum(losses) / len(losses)
            return {{"loss": loss, "perplexity": float(math.exp(min(loss, 20.0))), "examples": len(losses)}}

        def improvement_verdict(base_metrics, adapter_metrics, threshold_pct):
            base_loss = base_metrics.get("loss")
            adapter_loss = adapter_metrics.get("loss")
            warnings = []
            if base_loss is None or adapter_loss is None:
                return {{
                    "verdict": "inconclusive",
                    "reason": "Held-out loss was not available for both base and adapter models.",
                    "warnings": warnings,
                }}
            if base_metrics.get("examples", 0) < 3 or adapter_metrics.get("examples", 0) < 3:
                warnings.append("Very small eval sample count; treat this as a smoke signal, not proof.")
            delta = adapter_loss - base_loss
            pct = ((base_loss - adapter_loss) / base_loss * 100.0) if base_loss else 0.0
            if pct >= threshold_pct:
                verdict = "improved"
                reason = f"Adapter held-out loss improved by {{pct:.2f}}%."
            elif pct <= -threshold_pct:
                verdict = "regressed"
                reason = f"Adapter held-out loss worsened by {{abs(pct):.2f}}%."
            else:
                verdict = "inconclusive"
                reason = f"Loss delta {{pct:.2f}}% is below the {{threshold_pct:.2f}}% threshold."
            return {{
                "verdict": verdict,
                "reason": reason,
                "base_loss": base_loss,
                "adapter_loss": adapter_loss,
                "delta_loss": delta,
                "delta_loss_pct": pct,
                "threshold_pct": threshold_pct,
                "warnings": warnings,
            }}

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
            parser.add_argument("--eval-split", default="test")
            parser.add_argument("--eval-samples", type=int, default=20)
            parser.add_argument("--max-new-tokens", type=int, default=128)
            parser.add_argument("--improvement-threshold-pct", type=float, default=1.0)
            parser.add_argument("--stop-sequence", dest="stop_sequences", action="append", default=None)
            parser.add_argument("--fp16", action="store_true")
            parser.add_argument("--bf16", action="store_true")
            parser.add_argument("--verifier-path", default=None)
            args = parser.parse_args()
            stop_sequences = args.stop_sequences or ["### Human:", "\\n### Human:"]

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print("[rl-intern] loading dataset", flush=True)
            full_train_dataset = load_dataset(args.dataset, split=args.split)
            dataset = full_train_dataset
            if args.max_samples and len(dataset) > args.max_samples:
                dataset = dataset.select(range(args.max_samples))
            eval_source = args.eval_split
            eval_start_index = None
            try:
                eval_dataset = load_dataset(args.dataset, split=args.eval_split)
            except Exception as exc:
                eval_source = f"holdout_from_{{args.split}}"
                eval_start_index = min(args.max_samples or 0, len(full_train_dataset))
                eval_end_index = min(eval_start_index + args.eval_samples, len(full_train_dataset))
                eval_dataset = full_train_dataset.select(range(eval_start_index, eval_end_index))
                print(f"[rl-intern] eval split unavailable, using train holdout: {{exc}}", flush=True)
            if args.eval_samples and len(eval_dataset) > args.eval_samples:
                eval_dataset = eval_dataset.select(range(args.eval_samples))
            eval_rows = [eval_dataset[i] for i in range(len(eval_dataset))]
            eval_texts = [text for text in (row_text(row) for row in eval_rows) if text]
            eval_prompts = [prompt for prompt in (row_prompt(row) for row in eval_rows) if prompt]
            if not eval_prompts:
                eval_prompts = DEFAULT_PROMPTS
            eval_prompts = eval_prompts[:3]
            eval_info = {{
                "source": eval_source,
                "requested_samples": args.eval_samples,
                "actual_samples": len(eval_rows),
                "text_examples": len(eval_texts),
                "prompt_examples": len(eval_prompts),
                "train_rows": len(dataset),
                "train_max_samples": args.max_samples,
                "holdout_start_index": eval_start_index,
                "overlap_with_train": False if eval_start_index is not None else None,
            }}
            (output_dir / "eval_dataset_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
            print(f"[rl-intern] dataset ready: rows={{len(dataset)}} columns={{dataset.column_names}}", flush=True)
            print("[rl-intern] loading tokenizer", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("[rl-intern] loading model", flush=True)
            torch_dtype = torch.float16 if args.fp16 else "auto"
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=torch_dtype,
            )
            print("[rl-intern] model loaded", flush=True)
            print("[rl-intern] evaluating base model", flush=True)
            base_metrics = eval_loss(model, tokenizer, eval_texts)
            base_samples = generate_samples(model, tokenizer, eval_prompts, args.max_new_tokens, stop_sequences)
            (output_dir / "base_generations.json").write_text(
                json.dumps({{"status": "completed", "samples": base_samples}}, indent=2),
                encoding="utf-8",
            )
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
            print("[rl-intern] evaluating adapter model", flush=True)
            adapter_metrics = eval_loss(trainer.model, tokenizer, eval_texts)
            samples = generate_samples(trainer.model, tokenizer, eval_prompts, args.max_new_tokens, stop_sequences)
            eval_metrics = {{
                "status": "completed",
                "method": "{method}",
                "base": base_metrics,
                "adapter": adapter_metrics,
            }}
            evidence = improvement_verdict(base_metrics, adapter_metrics, args.improvement_threshold_pct)
            (output_dir / "adapter_generations.json").write_text(
                json.dumps({{"status": "completed", "samples": samples}}, indent=2),
                encoding="utf-8",
            )
            (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
            (output_dir / "improvement_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            (output_dir / "sample_generations.json").write_text(
                json.dumps({{"status": "completed", "samples": samples}}, indent=2)
            )
            print("[rl-intern] training completed", flush=True)

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
