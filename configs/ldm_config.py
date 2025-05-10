from dataclasses import dataclass


@dataclass
class LDMDatasetConfig:
    """
    Configuration class for the LDM dataset settings.
    """

    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"

    splits_root: str = "charsets"
    split_ratios: tuple[float, float] = (0.8, 0.2)
    random_seed: int = 2025
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class LDMModelConfig:
    """
    Configuration class for the LDM architecture settings.
    """

    unet_base_channels: int = 64

    time_pos_dim: int = 256
    time_emb_dim: int = 1024
    time_steps: int = 1000


@dataclass
class LDMTrainingConfig:
    """
    Configuration class for the LDM training settings.
    """

    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    num_epochs: int = 250

    pretrained_vqvae_path: str = "checkpoints/vqvae.pth"
    model_save_path: str = "checkpoints/ldm.pth"

    tensorboard_log_dir: str = "runs/LDM"

    sample_root: str = "samples"
    train_split: str = "train"
    val_split: str = "val"
    gt_split: str = "eval_outputs/gt"
    gen_split: str = "eval_outputs/gen"

    sample_steps: int = 50

    img_save_interval: int = 5
    lpips_eval_interval: int = 10
    eval_batch_size: int = 2


@dataclass
class LDMInferenceConfig:
    """
    Configuration class for the LDM inference settings.
    """

    pretrained_ldm_path: str = "checkpoints/ldm.pth"

    sample_root: str = "samples"
    ref_split: str = "inference/ref"
    gt_split: str = "inference/gt"
    gen_split: str = "inference/gen"

    batch_size: int = 16
    sample_steps: int = 50
