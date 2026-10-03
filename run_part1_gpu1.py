from pathlib import Path

import numpy as np
import torch

GPU_INDEX = 1
DEVICE = torch.device(f"cuda:{GPU_INDEX}")

torch.cuda.set_device(GPU_INDEX)
torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)

from attention import ForgetfulAttention

D = 3
V = torch.eye(D)
N_TRIALS = 1500
NS = [5, 10, 50, 100, 500, 1000, 5000]
SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "ranks_case1.npy"


def compute_ranks_part1() -> np.ndarray:
    if OUTPUT_PATH.exists():
        return np.load(OUTPUT_PATH)

    torch.manual_seed(SEED)
    Q = torch.eye(D)
    K = torch.eye(D)
    ranks = np.zeros((len(NS), N_TRIALS))

    for i, token_count in enumerate(NS):
        for trial in range(N_TRIALS):
            attention = ForgetfulAttention(
                K,
                Q,
                V,
                X=torch.rand(D, token_count) * 10 - 5,
            )
            attention.rescaled_dynamics()
            ranks[i, trial] = torch.linalg.matrix_rank(attention.P).item()
        print(f"Part 1: completed N={token_count} on {DEVICE}", flush=True)

    np.save(OUTPUT_PATH, ranks)
    return ranks


if __name__ == "__main__":
    print(f"Part 1 using {DEVICE}: {torch.cuda.get_device_name(GPU_INDEX)}")
    compute_ranks_part1()
