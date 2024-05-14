import os
import json

import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import barrier
from torchvision.datasets import FakeData

CUDA = torch.cuda.is_available()
BACKEND = "nccl" if CUDA else "gloo"

DEFAULT_CONFIG = {
    "epochs": 1,
    "batch_size": 32,
    "lr": 1e-3,
}
wandb.init(config=DEFAULT_CONFIG)
with open("/opt/ml/input/config/resourceconfig.json", "r") as f:
    resource_config = json.load(f)

# Setting up distributed training variables
NUM_GPUS = torch.cuda.device_count()
WORLD_SIZE = int(NUM_GPUS) * len(resource_config.get("hosts"))
RANK = resource_config.get("hosts").index(resource_config.get("current_host"))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

# Set the MASTER_ADDR and MASTER_PORT for torch.distributed
os.environ['MASTER_ADDR'] = resource_config.get("hosts")[0]
os.environ['MASTER_PORT'] = '7777'

# print some info
print("CUDA Available: ", CUDA)
print("WORLD_SIZE: ", WORLD_SIZE)
print("RANK: ", RANK)
print("LOCAL_RANK: ",LOCAL_RANK)
print("DEVICE: ", DEVICE)

def mnist_train():
    """Run training on the mnist dataset."""
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = FakeData(
        size=1000, image_size=(3, 128, 128), num_classes=196, transform=transforms
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(49152, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 196),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr)

    if CUDA:
        print("Moving model to GPU:", DEVICE)
        model = model.to(DEVICE)

    model = DDP(model)

    sampler = DistributedSampler(
        dataset,
        num_replicas=int(WORLD_SIZE),
        rank=int(RANK),
    )

    print("Batch size:", wandb.config.batch_size)
    loader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=sampler)

    barrier()

    for epoch in range(wandb.config.epochs):
        sampler.set_epoch(epoch)
        for _, (data, target) in enumerate(loader):
            if CUDA:
                data = data.to(DEVICE)
                target = target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data.view(data.shape[0], -1))
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})


if __name__ == "__main__":
    # Initialize the process group
    init_process_group(backend=BACKEND, rank=RANK, world_size=WORLD_SIZE)
    mnist_train()
    destroy_process_group()