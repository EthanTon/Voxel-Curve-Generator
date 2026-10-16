from dataclasses import replace

from ...Parser.spec import FeaturePathSpec
from .parser import polygon, read_ownership
from .spec import BindingSpec, FormationRule
from .values import check_keys, integer, names


def apply_rules(data, branches, transitions):
    definitions, bindings, ownership, formations = {}, [], [], []
    for rule in data.get("composition_rule", ()):
        check_keys(rule, "name handler source target owner boundary heights domains priority", "composition_rule")
        name = rule.get("name")
        if not isinstance(name, str) or not name or name in definitions:
            raise ValueError("composition rule names must be nonempty and unique")
        if rule.get("handler") not in {"bind", "share", "reserve", "formation"}:
            raise ValueError("composition rule handler must be bind, share, reserve, or formation")
        allowed = {"bind": "name handler source target", "share": "name handler source target owner",
                   "reserve": "name handler owner boundary heights domains priority",
                   "formation": "name handler target boundary heights"}
        check_keys(rule, allowed[rule["handler"]], "composition_rule")
        definitions[name] = rule
    for application in data.get("rule_application", ()):
        check_keys(application, "rule scope branches priority", "rule_application")
        if application.get("rule") not in definitions:
            raise ValueError("rule application references an unknown rule")
        rule = definitions[application["rule"]]
        if "priority" in application and rule["handler"] != "reserve":
            raise ValueError("rule_application.priority applies to reservations")
        scope = application.get("scope", "build")
        if scope != "build" and scope not in transitions:
            raise ValueError("rule application scope must be build or a transition name")
        selected = names(application.get("branches", list(branches)), "rule_application.branches")
        if not selected or set(selected) - branches.keys():
            raise ValueError("rule application needs known branches")
        if scope != "build" and set(selected) - transitions[scope]["branches"]:
            raise ValueError("rule application branches must participate in its transition")
        if rule["handler"] == "formation":
            kind, _, name = rule.get("target", "").partition(":")
            if kind != "formation" or not name:
                raise ValueError("formation rules need a formation:name target from the root build")
            components = tuple(component for branch in selected for component, _ in branches[branch])
            formation = components[0].build.formation(name)
            if any(component.build.formation(name) != formation for component in components):
                raise ValueError("shared formation definitions must match")
            region = {} if scope == "build" else transitions[scope]
            boundary = rule.get("boundary", region.get("boundary"))
            heights = rule.get("heights", region.get("heights"))
            if scope != "build" and boundary is None:
                raise ValueError("a transition formation needs a boundary")
            if heights is not None:
                if not isinstance(heights, (list, tuple)) or len(heights) != 2:
                    raise ValueError("formation heights need two integers")
                heights = tuple(integer(v, "formation heights") for v in heights)
                if heights[0] > heights[1]:
                    raise ValueError("formation heights must be ordered")
            formations.append(FormationRule(formation, tuple(c.name for c in components),
                              () if boundary is None else polygon(boundary, "formation.boundary"), heights))
            continue
        if rule["handler"] == "reserve":
            region = {} if scope == "build" else transitions[scope]
            options = {key: rule.get(key, region.get(key)) for key in ("owner", "boundary", "heights", "domains", "priority")}
            options = {key: value for key, value in options.items() if value is not None}
            options["priority"] = application.get("priority", options.get("priority", 0))
            ownership.extend(reservations(options, selected, branches))
            continue
        if scope != "build":
            raise ValueError("bind and share rules use build scope; crop a branch to limit their extent")
        target, source = rule.get("target", ""), rule.get("source")
        kind, _, name = target.partition(":")
        if kind not in {"structure", "wire"} or not name or not isinstance(source, str) or not source:
            raise ValueError("bind/share rules need a source path and structure:name or wire:name target")
        owner = rule.get("owner")
        if rule["handler"] == "share" and owner not in selected:
            raise ValueError("share owner must be one of its application branches")
        attribute = "structures" if kind == "structure" else "wires"
        for branch in selected:
            updated = []
            for component, run in branches[branch]:
                build = component.build
                items = getattr(build, attribute)
                if not any(item.name == name for item in items):
                    raise ValueError(f"{branch} has no {target}")
                if kind == "wire" and any(item.name == name and item.between for item in items):
                    raise ValueError("composition bindings need a longitudinal wire, not a between wire")
                if rule["handler"] == "share" and branch != owner:
                    component = replace(component, suppressed=(*component.suppressed, target))
                else:
                    placeholder = f"__composition_{target}"
                    options = {"wire": None} if kind == "structure" else {}
                    items = tuple(replace(item, feature_path=placeholder, spacing=0, angle=0, min_distance=0, **options)
                                  if item.name == name else item for item in items)
                    build = replace(build, **{attribute: items},
                                    feature_paths=(*build.feature_paths, FeaturePathSpec(placeholder, "path")))
                    bindings.append(BindingSpec(component.name, f"feature_path:{placeholder}", source))
                updated.append((replace(component, build=build), run))
            branches[branch] = updated
    return tuple(bindings), tuple(ownership), tuple(formations)


def reservations(options, selected, branches):
    owner = options.get("owner")
    if owner not in selected:
        raise ValueError("reservation owner must be one of its selected branches")
    owners = branches[owner]
    if len(owners) != 1:
        raise ValueError("a reservation owner must be a continuous branch")
    components = [component.name for branch in selected for component, _ in branches[branch]]
    return (read_ownership({**options, "owner": owners[0][0].name, "components": components}),)
