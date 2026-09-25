from pathlib import Path
from omegaconf import OmegaConf
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" 
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

def load_config(arguments=None):
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML configuration that overrides configs/default.yaml. "
            "Example: --config PSO.yaml"
        ),
    )

    args, overrides = parser.parse_known_args(arguments)

    if not DEFAULT_CONFIG.is_file():
        raise FileNotFoundError(
            f"Default configuration file not found: {DEFAULT_CONFIG}"
        )

    default_cfg = OmegaConf.load(DEFAULT_CONFIG)

    # All valid configuration fields must exist in default.yaml.
    OmegaConf.set_struct(default_cfg, True)

    if args.config is not None:
        override_config_path = CONFIG_DIR / args.config

        if not override_config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {override_config_path}"
            )

        file_cfg = OmegaConf.load(override_config_path)
    else:
        file_cfg = OmegaConf.create()

    # Convert:
    #     --angle 45 --tx 3
    # into:
    #     angle=45 tx=3
    cli_cfg = OmegaConf.from_dotlist(
        _to_dotlist(overrides)
    )

    cfg = OmegaConf.merge(
        default_cfg,
        file_cfg,
        cli_cfg,
    )

    return cfg

def store_config(
    angle,
    hx,
    hy,
    sx,
    sy,
    tx,
    ty,
    k1,
    k2,
    k3,
    p1,
    p2,
    config_name="PSO",
):
    # Prevent directories from being included in the filename.
    config_name = Path(config_name).name

    # Allow either "PSO" or "PSO.yaml".
    if config_name.lower().endswith((".yaml", ".yml")):
        config_name = Path(config_name).stem

    if not config_name:
        raise ValueError("config_name cannot be empty.")

    config_path = CONFIG_DIR / f"{config_name}.yaml"

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = OmegaConf.create({
        "angle": float(angle),
        "hx": float(hx),
        "hy": float(hy),
        "sx": float(sx),
        "sy": float(sy),
        "tx": float(tx),
        "ty": float(ty),
        "k1": float(k1),
        "k2": float(k2),
        "k3": float(k3),
        "p1": float(p1),
        "p2": float(p2),
    })

    OmegaConf.save(
        config=config,
        f=config_path,
    )

    return config_path

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