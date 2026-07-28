"""Utilities for config metadata tracking to detect parameter changes."""

import json
from pathlib import Path
from typing import Any, Dict, List


def get_config_metadata_path(output_file: Path) -> Path:
    """Return path to config metadata JSON for an output file."""
    return output_file.parent / f".config_{output_file.stem}.json"


def save_config_metadata(output_file: Path, config: Dict[str, Any], relevant_keys: List[str]) -> None:
    """Save relevant config parameters to metadata file."""
    relevant_config = {k: config[k] for k in relevant_keys if k in config}
    metadata_file = get_config_metadata_path(output_file)
    with open(metadata_file, 'w') as f:
        json.dump(relevant_config, f, indent=2)


def config_changed(output_file: Path, current_config: Dict[str, Any], relevant_keys: List[str]) -> bool:
    """Check if config has changed since file was created. Returns True if reprocessing needed."""
    metadata_file = get_config_metadata_path(output_file)
    
    if not metadata_file.exists():
        return True
    
    try:
        with open(metadata_file) as f:
            stored_config = json.load(f)
        current_relevant = {k: current_config[k] for k in relevant_keys if k in current_config}
        return current_relevant != stored_config
    except Exception:
        return True
