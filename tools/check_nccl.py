"""Check that multi-GPU gradient averaging (NCCL all-reduce) returns correct numbers on this machine.

Some PCIe setups silently corrupt GPU-to-GPU (P2P) copies, and a corrupted all-reduce shows up in training as random
huge or NaN gradients while every GPU's own loss looks normal. Each process fills a tensor from a known seed; after
the all-reduce every process compares the result with the same sum computed locally (with 2 GPUs it is exact).
  torchrun --standalone --nproc_per_node 2 tools/check_nccl.py              # as DDP uses it (P2P if available)
  NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node 2 tools/check_nccl.py   # through host memory instead
"""
import argparse
import math
import os

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--mb", type=int, default=25, help="tensor size in MB (DDP's default bucket is 25 MB)")
    args = ap.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    n = args.mb * 2 ** 20 // 4
    wrong, worst = 0, 0.0
    for it in range(args.iters):
        parts = [torch.randn(n, device="cuda", generator=torch.Generator("cuda").manual_seed(it * 1000 + r))
                 for r in range(world)]
        x = parts[rank].clone()
        dist.all_reduce(x)
        err = (x - torch.stack(parts).sum(0)).abs().max().item()
        if not err <= 1e-5:  # also catches NaN
            wrong += 1
            print(f"[rank {rank}] iteration {it}: max abs error {err}", flush=True)
        worst = err if math.isnan(err) else max(worst, err)
    print(f"[rank {rank}] {args.iters} all-reduces of {args.mb} MB over {world} GPUs: {wrong} wrong, "
          f"max abs error {worst} (P2P disabled: {os.environ.get('NCCL_P2P_DISABLE', '0') == '1'})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
