import json
from hashlib import md5


def _serialize_for_hash(v):
    """Produce a JSON-serializable value for hashing."""
    if isinstance(v, (str, int, float)):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, list):
        return sorted(v) if all(isinstance(x, (str, int, float)) for x in v) else v
    return None


class Hashable:
    @property
    def hash(self):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, tuple)):
                d[k] = list(v) if isinstance(v, tuple) else v
            elif isinstance(v, list):
                d[k] = _serialize_for_hash(v)
            else:
                continue
        return md5(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()
