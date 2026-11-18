"""Does set_per_process_memory_fraction stop an allocation under this allocator? Run in .venvs/seedvr2."""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = sys.argv[1]
import torch

total = torch.cuda.mem_get_info()[1]
torch.cuda.set_per_process_memory_fraction(0.25)
try:
    x = torch.empty(int(total * 0.4), dtype=torch.uint8, device="cuda")
    print(f"{sys.argv[1]}: NOT enforced, allocated {x.numel() / 1024**3:.1f} GB")
except torch.OutOfMemoryError:
    print(f"{sys.argv[1]}: enforced")
