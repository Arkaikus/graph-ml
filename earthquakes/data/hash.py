import json
from hashlib import md5


class Hashable:

    @property
    def hash(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_") if isinstance(v, (str, int, float, tuple))}
        return md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
