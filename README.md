# distributed-pytorch-training
This repo demonstrates distributed training with torch.nn.parallel.DistributedDataParallel on multiple GPUs.

## Project Overview

“This repo demonstrates distributed training with torch.nn.parallel.DistributedDataParallel on multiple GPUs, featuring: mixed precision, gradient sync profiling, and scaling experiments.”

## Architecture

- How processes are spawned.
- How DDP is used (process group, sampler, etc).
- How AMP is integrated.
- How metrics are collected.

## How to Run

- Environment (PyTorch version, CUDA, etc.).

- Commands for:
-> Single GPU baseline.
-> 2-GPU DDP.
-> 4-GPU DDP.
-> Profiling run.

## Experiments & Results

Tables + plots:

- Single vs multi-GPU comparison.
- Throughput vs batch size.
- AMP vs FP32.

Brief analysis of bottlenecks:

- e.g., “At 4 GPUs, gradient synchronization accounts for ~X% of step time.”
