import shutil
import random
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm
import numpy as np

# TODO: remove if redundant
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image






import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from diffusers.models import AutoencoderKL
from dnn import create_dnn
from flow import FlowMatching

import wandb


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file.
    """

    if dist.get_rank() == 0: # master process
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def ddp_setup(global_seed: int):
    """
    DDP init for torchrun
    returns: rank, local_rank, world_size, device
    """

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int (os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

    seed = global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return rank, local_rank, world_size, device

def main(args):
    """
    Approximate a flow field instrumenting a DNN
    """

    assert torch.cuda.is_available(), f"Training requires at least one GPU currently."

    rank, local_rank, world_size, device = ddp_setup(args.global_seed)


    # Setup an experiment folder:
    experiment_index = args.exp
    experiment_dir = f"{args.results_dir}/{experiment_index}" # create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints" # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        mode = args.use_wandb
        wandb.init(
            project = "FlowVideo",
            entity = "red-blue-purple",
            config=vars(args),
            name=experiment_index,
            mode=mode
        )
        print ("W&B run id: ", wandb.run.id)
        print("W&B url:", wandb.run.url)
    else:
        logger = create_logger(None)
    
    # Create model + dnn
    assert args.spatial_resolution % 8 == 0, "Resolution must be divisible by 8 (for the VAE)"
    latent_res = args.spatial_resolution // 8

    DNN = create_dnn(args)

    DNN = DDP(DNN.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    flow = FlowMatching(DNN, t_sampler=args.t_sampler)
    


# exp (name)
# results_dir (results)
# global seed
# use_wandb
# spatial_resolution






