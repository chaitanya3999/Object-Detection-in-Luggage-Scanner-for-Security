"""
Configuration loader for the X-Ray Object Detection project.
Loads YAML config files and provides easy access to parameters.
"""

import os
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config YAML file. If None, loads configs/default.yaml
                     relative to the project root.

    Returns:
        Dictionary with all configuration parameters.
    """
    if config_path is None:
        # Find project root (directory containing 'configs/')
        project_root = _find_project_root()
        config_path = os.path.join(project_root, "configs", "default.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def _find_project_root() -> str:
    """
    Find the project root by looking for the 'configs' directory.
    Walks up from the current file's location.
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Walk up at most 5 levels
        if (current / "configs").exists():
            return str(current)
        current = current.parent

    # Fallback: use current working directory
    return os.getcwd()


def get_property_names(config: dict) -> list:
    """Get list of property names from the config schema."""
    return [p["name"] for p in config["property_schema"]["properties"]]


def get_property_ranges(config: dict) -> dict:
    """Get property name -> (min, max) range mapping."""
    ranges = {}
    for p in config["property_schema"]["properties"]:
        if "range" in p:
            ranges[p["name"]] = tuple(p["range"])
    return ranges


def resolve_paths(config: dict, base_dir: str = None, use_colab: bool = False) -> dict:
    """
    Resolve relative paths in config to absolute paths.

    Args:
        config: Configuration dictionary.
        base_dir: Base directory for resolving relative paths.
        use_colab: If True, use Google Drive paths from config.

    Returns:
        Config with resolved paths.
    """
    if base_dir is None:
        base_dir = _find_project_root()

    paths = config.get("paths", {})

    if use_colab:
        # In Colab, use Drive-based paths for data and checkpoints
        resolved = {
            "data_root": paths.get("drive_data", "/content/drive/MyDrive/xray_detection/data"),
            "raw_data": os.path.join(paths.get("drive_data", ""), "raw"),
            "processed_data": os.path.join(paths.get("drive_data", ""), "processed"),
            "annotations": os.path.join(paths.get("drive_data", ""), "annotations"),
            "checkpoints": paths.get("drive_checkpoints", "/content/drive/MyDrive/xray_detection/checkpoints"),
            "results": os.path.join(paths.get("drive_root", ""), "results"),
        }
    else:
        # Local development
        resolved = {}
        for key in ["data_root", "raw_data", "processed_data", "annotations", "checkpoints", "results"]:
            if key in paths:
                p = paths[key]
                if not os.path.isabs(p):
                    p = os.path.join(base_dir, p)
                resolved[key] = p

    config["resolved_paths"] = resolved
    return config
