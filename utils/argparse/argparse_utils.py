import argparse
from typing import TypeVar

T = TypeVar("T")


def update_config_from_args(
    converting_config: T,
    args: argparse.Namespace,
) -> T:
    """
    Update the configuration object with values from the command line arguments.
    """
    for key, value in vars(args).items():
        if hasattr(converting_config, key) and (value is not None):
            if isinstance(value, tuple) or isinstance(value, list):
                setattr(
                    converting_config, key, type(getattr(converting_config, key))(value)
                )
            else:
                setattr(converting_config, key, value)
    return converting_config
