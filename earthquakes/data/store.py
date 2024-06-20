import json
import logging
import os
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

    def save(self, data, file_name: str):
        try:
            file_path = self.path / file_name
            self.logger.debug(f"Write cache data: {file_path}")
            if file_path.suffix == ".json":
                assert type(data) == dict, "[data] must be a dict"
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=4)
            elif file_path.suffix in [".yml", ".yaml"]:
                assert type(data) == dict, "[data] must be a dict"
                with open(file_path, "w") as f:
                    yaml.safe_dump(data, f, indent=4)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)

            return True
        except Exception as e:
            self.logger.exception(f"Saving failed: {e}")
            return False

    def load(self, file_name: str):
        try:
            file_path = self.path / file_name
            assert file_path.exists() and file_path.is_file()

            self.logger.debug(f"Loading cached data: {file_path}")
            if file_path.suffix == "json":
                with open(file_path, "r") as f:
                    return json.load(f)
            elif file_path.suffix in ["yml", "yaml"]:
                with open(file_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
        except:
            self.logger.debug(f"Loading failed")
            return None
