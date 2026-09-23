from pathlib import Path
from omegaconf import OmegaConf
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" 

def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Name of the YAML file in configs/",
    )

    args, overrides = parser.parse_known_args()

    config_path = CONFIG_DIR / f"{args.config}"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(_to_dotlist(overrides))

    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    return cfg

def _to_dotlist(arguments):
    # ex) Convert ['--angle', '45'] into ['angle=45']
    
    dotlist = []
    index = 0

    while index < len(arguments):
        argument = arguments[index]

        if not argument.startswith("--"):
            raise ValueError(
                f"Expected an argument beginning with '--', got: {argument}"
            )

        key = argument.removeprefix("--")

        if index + 1 >= len(arguments):
            raise ValueError(f"Missing value for --{key}")

        value = arguments[index + 1]
        dotlist.append(f"{key}={value}")
        index += 2

    return dotlist