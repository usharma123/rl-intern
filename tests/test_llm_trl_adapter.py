from rl_intern.domains.llm_trl.dataset import inspect_llm_dataset
from rl_intern.domains.llm_trl.scripts import build_trl_training_script
from rl_intern.domains.llm_trl.verifier import validate_grpo_verifier


def test_sft_dataset_accepts_messages_rows():
    result = inspect_llm_dataset(rows=[{"messages": [{"role": "user", "content": "hi"}]}], method="sft")

    assert result["valid"] is True
    assert result["format"] == "chat_messages"


def test_dpo_dataset_rejects_missing_rejected():
    result = inspect_llm_dataset(rows=[{"prompt": "p", "chosen": "c"}], method="dpo")

    assert result["valid"] is False
    assert "rejected" in result["reason"]


def test_grpo_verifier_accepts_numeric_score():
    source = "def score(example, completion):\n    return 1.0\n"

    result = validate_grpo_verifier(verifier_source=source)

    assert result["valid"] is True
    assert result["score"] == 1.0


def test_grpo_verifier_rejects_non_numeric_score():
    source = "def score(example, completion):\n    return 'good'\n"

    result = validate_grpo_verifier(verifier_source=source)

    assert result["valid"] is False


def test_generate_grpo_script_contains_reward_func():
    script = build_trl_training_script("grpo")

    assert "GRPOTrainer" in script
    assert "reward_func" in script
