from pathlib import Path
import yaml

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

def load_config(path=project_root / "configs" / "default.yaml"):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)