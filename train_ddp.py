# train_ddp.py

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training of ResNet-18 on CIFAR-10")

    parser.add_argument("--data-dir", default="./data", type=str,
                        help="directory to store CIFAR-10")
    parser.add_argument("--epochs", default=10, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch-size", default=128, type=int,
                        help="mini-batch size *per GPU*")
    parser.add_argument("--workers", default=4, type=int,
                        help="number of data loading workers per process")
    parser.add_argument("--lr", default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4, type=float,
                        help="weight decay")
    parser.add_argument("--amp", action="store_true",
                        help="use automatic mixed precision (AMP)")
    parser.add_argument("--print-freq", default=50, type=int,
                        help="print frequency (in steps)")
    parser.add_argument("--output-dir", default="./experiments", type=str,
                        help="directory to save logs")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")

    args = parser.parse_args()
    return args


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    """
    Initialize distributed process group (if WORLD_SIZE > 1) and return
    (rank, world_size, local_rank).
    """
    if not is_distributed():
        return 0, 1, 0

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def set_seed(seed, rank):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def get_dataloaders(args, rank, world_size):
    """
    Returns train_loader, val_loader, train_sampler.
    Uses DistributedSampler for training when in DDP.
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=(rank == 0),  # only rank 0 downloads
        transform=transform_train,
    )

    # Make sure other ranks wait until data is downloaded
    if is_distributed():
        dist.barrier()

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transform_test,
    )

    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler


def build_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    return model


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """
    Helper to all_reduce a scalar tensor across processes.
    If not distributed, returns the tensor unchanged.
    """
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    args,
    rank,
    world_size,
    scaler=None,
    train_sampler=None,
):
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    start_epoch = time.time()
    accum_time = 0.0
    accum_samples = 0

    for i, (images, targets) in enumerate(train_loader):
        batch_start = time.time()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if args.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        epoch_loss += loss.item() * batch_size

        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()
        epoch_correct += correct
        epoch_total += batch_size

        batch_time = time.time() - batch_start
        accum_time += batch_time
        accum_samples += batch_size * world_size  # global samples this step

        # Logging (rank 0 only)
        if rank == 0 and (i + 1) % args.print_freq == 0:
            avg_loss = epoch_loss / epoch_total
            avg_acc = 100.0 * epoch_correct / epoch_total
            if accum_time > 0:
                throughput = accum_samples / accum_time
            else:
                throughput = 0.0

            print(
                f"Epoch [{epoch+1}] Step [{i+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f}  "
                f"Acc: {avg_acc:.2f}%  "
                f"Throughput: {throughput:.2f} samples/s (global)"
            )
            accum_time = 0.0
            accum_samples = 0

    # Reduce metrics across processes
    loss_tensor = torch.tensor(epoch_loss, device=device)
    correct_tensor = torch.tensor(epoch_correct, device=device, dtype=torch.long)
    total_tensor = torch.tensor(epoch_total, device=device, dtype=torch.long)

    loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
    correct_tensor = reduce_tensor(correct_tensor, op=dist.ReduceOp.SUM)
    total_tensor = reduce_tensor(total_tensor, op=dist.ReduceOp.SUM)

    epoch_time = time.time() - start_epoch

    if rank == 0:
        global_loss = loss_tensor.item() / total_tensor.item()
        global_acc = 100.0 * correct_tensor.item() / total_tensor.item()
    else:
        global_loss, global_acc = None, None

    return global_loss, global_acc, epoch_time


@torch.no_grad()
def validate(model, val_loader, criterion, device, rank, world_size):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        val_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        val_correct += preds.eq(targets).sum().item()
        val_total += batch_size

    loss_tensor = torch.tensor(val_loss, device=device)
    correct_tensor = torch.tensor(val_correct, device=device, dtype=torch.long)
    total_tensor = torch.tensor(val_total, device=device, dtype=torch.long)

    loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
    correct_tensor = reduce_tensor(correct_tensor, op=dist.ReduceOp.SUM)
    total_tensor = reduce_tensor(total_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        global_loss = loss_tensor.item() / total_tensor.item()
        global_acc = 100.0 * correct_tensor.item() / total_tensor.item()
    else:
        global_loss, global_acc = None, None

    return global_loss, global_acc


def maybe_wrap_ddp(model, device, local_rank):
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    return model


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device}, world_size={world_size}, amp={args.amp}")

    torch.backends.cudnn.benchmark = True

    # Data
    train_loader, val_loader, train_sampler = get_dataloaders(args, rank, world_size)

    # Model / optimizer / loss
    model = build_model(device)
    model = maybe_wrap_ddp(model, device, local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # CSV log (rank 0 only)
    metrics_path = Path(args.output_dir) / "metrics_rank0.csv"
    if rank == 0 and not metrics_path.exists():
        with metrics_path.open("w") as f:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_seconds\n"
            )

    for epoch in range(args.epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args,
            rank,
            world_size,
            scaler=scaler,
            train_sampler=train_sampler,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device, rank, world_size)

        if rank == 0:
            print(
                f"[Epoch {epoch+1}/{args.epochs}] "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | "
                f"Epoch time: {epoch_time:.2f}s"
            )
            with metrics_path.open("a") as f:
                f.write(
                    f"{epoch+1},{train_loss:.4f},{train_acc:.2f},"
                    f"{val_loss:.4f},{val_acc:.2f},{epoch_time:.4f}\n"
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
