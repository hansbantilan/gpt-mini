import yaml


def load_params(path: str) -> dict:
    with open(path) as f:
        params = yaml.safe_load(f)
    return params
