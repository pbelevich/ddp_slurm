import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def run_worker(rank, world_size, args):
    print(f"rank = {rank} host/pid = {socket.gethostname()}/{os.getpid()}")
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    gpu = 0 # because ddp_slurm.sh sets CUDA_VISIBLE_DEVICES to SLURM_LOCALID

    print(f"rank = {rank} init_process_group")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"rank = {rank} starting")

    # create model and move it to GPU with id rank
    model = ToyModel().to(gpu)
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(gpu))
    labels = torch.randn(20, 5).to(gpu)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    dist.destroy_process_group()
    print(f"rank = {rank} exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDP NCCL SLURM experiment')
    parser.add_argument('--world_size', type=int, default=None,
                        help='the world size to initiate DPP')
    parser.add_argument('--rank', type=int, default=None,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help="""Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port.""")
    parser.add_argument('--master_port', type=str, default='29500',
                        help="""Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    if args.rank is None:
        pass
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")

