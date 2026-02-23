import json
import os


def parse_scalar(raw):
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_yaml_or_json_config(path):
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as file_obj:
        if suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(file_obj) or {}
        elif suffix == ".json":
            data = json.load(file_obj)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dictionary")
    return data


def load_section_config(config_path, section_name, overrides):
    all_cfg = load_yaml_or_json_config(config_path)
    merged = {}
    merged.update(all_cfg.get("global", {}))
    merged.update(all_cfg.get(section_name, {}))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}")
        key, value = item.split("=", 1)
        merged[key.strip()] = parse_scalar(value)
    return merged


def validate_required_nonempty(cfg, key, section):
    value = cfg.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{section}.{key} is required")
    return str(value).strip()
