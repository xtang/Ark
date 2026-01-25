"""Configuration loader for the podcast generator."""

import os
from pathlib import Path
from typing import Any

import yaml


def get_config_path() -> Path:
    """Get the path to the default config file."""
    # __file__ is src/config.py, so parent.parent gets us to podcast_generator/
    return Path(__file__).parent.parent / "config" / "default_config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Uses default if None.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if config_path is None:
        config_path = get_config_path()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute
    if "output" in config and "directory" in config["output"]:
        output_dir = config["output"]["directory"]
        if not os.path.isabs(output_dir):
            config["output"]["directory"] = str(
                config_path.parent.parent / output_dir
            )

    if "database" in config and "path" in config["database"]:
        db_path = config["database"]["path"]
        if not os.path.isabs(db_path):
            config["database"]["path"] = str(config_path.parent.parent / db_path)

    return config


def get_speakers(config: dict[str, Any]) -> dict[str, str]:
    """
    Get speaker name to voice ID mapping from config.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict mapping speaker names to ElevenLabs voice IDs.
    """
    speakers = config.get("dialogue", {}).get("speakers", [])
    return {s["name"]: s["voice_id"] for s in speakers}


def get_topic_name(config: dict[str, Any], topic_key: str) -> str:
    """
    Get the display name for a topic key.

    Args:
        config: Configuration dictionary.
        topic_key: Topic key (e.g., 'life_tips').

    Returns:
        Topic display name in Chinese.

    Raises:
        KeyError: If topic key not found.
    """
    topics = config.get("topics", {})
    if topic_key not in topics:
        raise KeyError(f"Unknown topic: {topic_key}. Available: {list(topics.keys())}")
    return topics[topic_key]
