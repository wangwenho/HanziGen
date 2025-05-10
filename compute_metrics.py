import argparse

from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import select_device
from utils.metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Compute metrics for generated images")
    parser.add_argument(
        "--generated_img_dir", type=str, help="Generated image directory"
    )
    parser.add_argument(
        "--ground_truth_img_dir", type=str, help="Ground Truth image directory"
    )
    parser.add_argument("--eval_batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, help="Evaluation device (mps, cpu, cuda)")

    return parser.parse_args()


def main() -> None:
    """ """
    args = parse_args()
    device = select_device(args.device)

    compute_all_metrics(
        generated_img_dir=args.generated_img_dir,
        ground_truth_img_dir=args.ground_truth_img_dir,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
