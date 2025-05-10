import torch
import torch.nn as nn


def select_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the optimal device for PyTorch operations.
    """
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def print_model_params(model: nn.Module) -> None:
    """
    Print the total number of parameters in a PyTorch model.
    """
    params = sum(p.numel() for p in model.parameters())

    if params >= 1e9:
        formatted_params = f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        formatted_params = f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        formatted_params = f"{params / 1e3:.2f}K"
    else:
        formatted_params = str(params)

    print(f"Total model parameters: {formatted_params}")
