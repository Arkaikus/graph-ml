import json
import logging
import pickle
from pathlib import Path

import yaml


class Store:
    """handler to store data to disk"""

    logger = logging.getLogger(__name__)

    def __init__(self, cache_name: str, parent="cache") -> None:
        self.cache_name = cache_name
        self.root = Path(parent)
        self.root.mkdir(exist_ok=True)
        self.path = self.root / cache_name
        self.path.mkdir(exist_ok=True)

    def save(self, data, file_name: str) -> bool:
        try:
            file_path = self.path / file_name
            self.logger.debug("Write cache data: %s", file_path)
            if file_path.suffix == ".json":
                if not isinstance(data, dict):
                    raise TypeError("data must be a dict for JSON output")
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=4)
            elif file_path.suffix in [".yml", ".yaml"]:
                if not isinstance(data, dict):
                    raise TypeError("data must be a dict for YAML output")
                with open(file_path, "w") as f:
                    yaml.safe_dump(data, f, indent=4)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
            return True
        except (OSError, TypeError) as e:
            self.logger.exception("Saving failed: %s", e)
            return False

    def load(self, file_name: str):
        file_path = self.path / file_name
        if not file_path.exists() or not file_path.is_file():
            self.logger.debug("No cache file at %s", file_path)
            return None
        try:
            self.logger.debug("Loading cached data: %s", file_path)
            if file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    return json.load(f)
            if file_path.suffix in [".yml", ".yaml"]:
                with open(file_path, "r") as f:
                    return yaml.safe_load(f)
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.warning("Loading failed for %s: %s", file_path, e)
            return None
