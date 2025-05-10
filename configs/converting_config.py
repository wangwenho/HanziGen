from dataclasses import dataclass


@dataclass
class ConvertingConfig:
    """
    Configuration class for converting settings.
    """

    blacklevel: float = 0.5
    turdsize: int = 2
    alphamax: float = 1
    opttolerance: float = 0.2
