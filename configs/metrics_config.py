from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """
    Configuration class for the metrics settings.
    """

    generated_img_dir = "samples/inference/gen"
    ground_truth_img_dir = "samples/inference/gt"

    eval_batch_size: int = 2
