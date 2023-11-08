import torch.distributed as dist
import os
import torch

def setup_distributed(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    if world_size != 1:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]