from agent.tools.env_inspect import inspect_env


def _normalize_algo(name: str | None) -> str | None:
    return name.upper() if name else None


def choose_algorithm(env_id: str, user_preference: str | None = None) -> dict:
    env_result = inspect_env(env_id)
    if env_result.get("error"):
        return {
            "env_id": env_id,
            "recommended_algorithm": None,
            "compatible_algorithms": [],
            "reason": "Environment inspection failed.",
            "warnings": [env_result["error"]],
        }

    action_type = env_result.get("action_space_type", "unknown")
    warnings = list(env_result.get("warnings", []))

    if action_type == "Discrete":
        compatible = ["PPO", "DQN"]
        recommended = "PPO"
        reason = (
            f"{env_id} has a discrete action space, so PPO is a robust default "
            "and DQN is also compatible."
        )
    elif action_type == "Box":
        compatible = ["PPO", "SAC", "TD3"]
        recommended = "PPO"
        reason = (
            f"{env_id} has a continuous Box action space, so PPO is a robust "
            "default while SAC and TD3 are also compatible."
        )
    elif action_type in {"MultiDiscrete", "MultiBinary"}:
        compatible = ["PPO"]
        recommended = "PPO"
        reason = f"{env_id} has a {action_type} action space; PPO is the v0.1 recommendation."
    else:
        compatible = []
        recommended = None
        reason = f"{env_id} has unknown action space type {action_type}; manual review is needed."
        warnings.append("Unknown action space type; cannot safely choose an algorithm.")

    requested = _normalize_algo(user_preference)
    if requested:
        if requested in compatible:
            recommended = requested
            reason = f"User requested {requested}, which is compatible with {action_type}."
        else:
            warnings.append(
                f"Requested algorithm {requested} may be incompatible with action space type {action_type}."
            )
            if requested in {"PPO", "DQN", "SAC", "TD3"}:
                recommended = requested

    return {
        "env_id": env_id,
        "recommended_algorithm": recommended,
        "compatible_algorithms": compatible,
        "reason": reason,
        "warnings": warnings,
    }
