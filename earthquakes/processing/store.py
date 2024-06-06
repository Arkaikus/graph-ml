import json
import logging
import os
import pickle

import yaml


class Store:
    logger = logging.getLogger(__name__)

    def __init__(self, cache_name, file_name, file_type="pkl", save_folder="cache") -> None:
        self.save_folder = save_folder
        self.cache_name = cache_name
        self.file_name = file_name
        self.file_type = file_type
        self.cache_folder = save_folder + "/" + cache_name
        os.makedirs(self.cache_folder, exist_ok=True)
        self.file_path = self.cache_folder + "/" + file_name + "." + file_type
        self.data = self.load()

    @property
    def empty(self):
        return self.data is None

    def save(self, data):
        try:
            self.logger.debug(f"Write cache data: {self.file_path}")
            if self.file_type == "pkl":
                with open(self.file_path, "wb") as f:
                    pickle.dump(data, f)
            elif self.file_type == "json":
                assert type(data) == dict, "[data] must be a dict"
                with open(self.file_path, "w") as f:
                    json.dump(data, f, indent=4)
            elif self.file_type in ["yml", "yaml"]:
                assert type(data) == dict, "[data] must be a dict"
                with open(self.file_path, "w") as f:
                    yaml.safe_dump(data, f, indent=4)
            else:
                raise NotImplementedError

            return True
        except Exception as e:
            self.logger.error(f"Saving failed: {e}")
            return False

    def load(self):
        try:
            self.logger.debug(f"Loading cached data: {self.file_path}")
            if self.file_type == "pkl":
                with open(self.file_path, "rb") as f:
                    return pickle.load(f)
            elif self.file_type == "json":
                with open(self.file_path, "r") as f:
                    return json.load(f)
            elif self.file_type in ["yml", "yaml"]:
                with open(self.file_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                raise NotImplementedError
        except:
            self.logger.debug(f"Loading failed")
            return None
