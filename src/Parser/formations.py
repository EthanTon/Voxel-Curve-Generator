from .spec import FormationSpec
from .values import require


def read_formation(data):
    keys = "name domain rail_domains replace_domains foundation_blocks support flank shoulder edge clearance edge_height".split()
    if set(data) - set(keys):
        raise ValueError("formation has unknown options")
    values = {key: require(data, key) for key in keys[:9]}
    for key in ("rail_domains", "replace_domains", "foundation_blocks"):
        value = values[key]
        if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
            raise ValueError(f"formation.{key} needs a nonempty list of names")
        values[key] = tuple(value)
    for key in ("name", "domain", "support", "flank", "shoulder", "edge"):
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"formation.{key} needs a nonempty string")
    for key in ("clearance", "edge_height"):
        value = data.get(key, 2)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"formation.{key} needs a positive integer")
        values[key] = value
    if set(values["rail_domains"]) & set(values["replace_domains"]):
        raise ValueError("formation must preserve its rail domains")
    if values["domain"] in (*values["rail_domains"], *values["replace_domains"]):
        raise ValueError("resolved formation needs its own domain")
    return FormationSpec(**values)
