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
    speakers_map = {}
    
    def add(source):
        if isinstance(source, dict):
             for lang_s in source.values():
                 if isinstance(lang_s, list):
                     for s in lang_s:
                         speakers_map[s["name"]] = s["voice_id"]
        elif isinstance(source, list):
             for s in source:
                 speakers_map[s["name"]] = s["voice_id"]

    # Global
    add(config.get("dialogue", {}).get("speakers", []))
    
    # Topics
    for topic_conf in config.get("topics", {}).values():
        if isinstance(topic_conf, dict):
            add(topic_conf.get("speakers"))
            
    return speakers_map


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
    
    val = topics[topic_key]
    if isinstance(val, dict):
        return val.get("name", topic_key)
    return val


def get_topic_config(config: dict[str, Any], topic_key: str) -> dict[str, Any]:
    """
    Get the full configuration for a specific topic.
    
    Args:
        config: Global configuration dictionary.
        topic_key: Topic key.
        
    Returns:
        Dictionary containing topic-specific settings (prompt, model, tools, etc.)
        merged with global defaults where applicable.
    """
    topics = config.get("topics", {})
    topic_val = topics.get(topic_key, {})
    
    # Normalize to dict if it's just a string (old format)
    if isinstance(topic_val, str):
        topic_conf = {"name": topic_val}
    else:
        topic_conf = topic_val.copy()
        
    # Get global defaults for fallback
    dialogue_defaults = config.get("dialogue", {})
    
    # 1. Prompt template key
    # Default to topic_key if not specified, or 'default' if that doesn't exist?
    # Actually dialogue generator creates the prompt from template.
    # We just pass the config.
    
    # Ensure 'name' is set
    if "name" not in topic_conf and isinstance(topic_val, str):
         topic_conf["name"] = topic_val
         
    return topic_conf


def get_prompts_path() -> Path:
    """Get the path to the prompts config file."""
    return Path(__file__).parent.parent / "config" / "prompts.yaml"


def load_prompts(prompts_path: Path | str | None = None) -> dict[str, str]:
    """
    Load prompt templates from YAML file.

    Args:
        prompts_path: Path to prompts file. Uses default if None.

    Returns:
        Dictionary mapping prompt names to template strings.
    """
    if prompts_path is None:
        prompts_path = get_prompts_path()

    prompts_path = Path(prompts_path)

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    return prompts or {}
