import math
from pathlib import Path

import torch


def calculate_adaptive_wd(
    dataset_size: int, batch_size: int, epsilon: float = 1e-3
) -> float:
    if dataset_size <= 0:
        raise ValueError("Dataset size must be positive.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    steps_per_epoch = math.ceil(dataset_size / batch_size)
    if steps_per_epoch == 0:
        return 0.0

    adaptive_lambda = 1 - math.pow(epsilon, 1 / steps_per_epoch)
    return adaptive_lambda

def get_dataset_size_from_tensor(file_path: Path) -> int:
    if not file_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {file_path}")
    tensor = torch.load(file_path)
    return tensor.size(0)


if __name__ == "__main__":
    try:
        data_path = Path("./data/cache/wikitext2_train_line_pack_ids_v3.pt")

        wikitext2_train_size = get_dataset_size_from_tensor(data_path)
        batch_size = 8

        adaptive_wd = calculate_adaptive_wd(wikitext2_train_size, batch_size)

        print(f"Successfully loaded preprocessed data from: {data_path}")
        print(f"Actual Wikitext-2 Training Set Size: {wikitext2_train_size} sequences")
        print(f"Batch Size: {batch_size}")
        print(f"Steps per Epoch: {math.ceil(wikitext2_train_size / batch_size)}")
        print(f"Calculated Adaptive Weight Decay (λ*): {adaptive_wd:.8f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the wikitext-2 dataset has been preprocessed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
