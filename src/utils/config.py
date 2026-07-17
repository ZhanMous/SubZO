"""Configuration loading and validation."""

import json
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a JSON configuration file.

    Args:
        path: Path to JSON config file

    Returns:
        Configuration dict
    """
    with open(path) as f:
        return json.load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two config dicts. Override takes precedence.

    Args:
        base: Base configuration
        override: Override values

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict) -> list[str]:
    """Validate a configuration dict. Returns list of error messages (empty = valid).

    Args:
        config: Configuration to validate

    Returns:
        List of error messages
    """
    errors = []

    required_top = ["dataset", "model", "time_steps", "num_classes"]
    for key in required_top:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    if "base_session" in config:
        bs = config["base_session"]
        for key in ["num_classes", "epochs", "batch_size", "lr"]:
            if key not in bs:
                errors.append(f"Missing base_session.{key}")

    if "incremental_sessions" in config:
        inc = config["incremental_sessions"]
        for key in ["num_sessions", "ways", "shots"]:
            if key not in inc:
                errors.append(f"Missing incremental_sessions.{key}")

    return errors
