import torch
import argparse
import torch.distributed as dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--local_rank", type=int)
    opts = parser.parse_args()
    torch.cuda.set_device(opts.local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://")

