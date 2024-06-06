import json
from hashlib import md5


class Hashable:

    @property
    def hash(self):
        data = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            if not callable(v)
            if not isinstance(v, (dict, property))
        }
        return md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
