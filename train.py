import os
import sys
import wandb
import shutil
import argparse
from functools import partial

import random
import numpy as np
from datetime import datetime

import torch
# Set TORCH_CUDA_ARCH_LIST before importing torch to avoid compilation warnings
# This should match your GPU architecture (8.9 for RTX 4090)
# More available in: https://developer.nvidia.com/cuda-gpus
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"

import deepspeed
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools.trainer import train, validate
from src.models.hammer import Hammer
from src.utils.logger import get_root_logger
from src.utils.afford_dataset import PIADv1Afford, PIADAfford, collate_fn


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def parse_args(args):
    parser = argparse.ArgumentParser(description="Affordance Reasoning Model Training")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--model_max_length", default=8192, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--log_base_dir", default="./exp", type=str)
    parser.add_argument("--project_name", default="3DAffordance", type=str)
    parser.add_argument("--exp_name", default="PIAD", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="validation batch size per device"
    )
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj, v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--vlm_out_dim", default=256, type=int)
    parser.add_argument("--dataset", default="PIADv2", type=str, choices=["PIADv1", "PIADv2", "HYBRID"])
    parser.add_argument("--question_type", default="simple", type=str)
    parser.add_argument("--setting", default="Seen", type=str, choices=["Seen", "Unseen", "Unseen_afford", "Unseen_obj"])
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch (overridden if resuming)")
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    seed = set_seed(args.seed)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        log_file = os.path.join(args.log_dir, f'train.log')
        logger = get_root_logger(log_file=log_file, name='train')
        logger.info(f"Logging to {log_file}")
        logger.info(f"Training arguments: {args}")
        logger.info(f"Using seed: {seed}")

        if args.debug:
            wandb.init(mode="disabled")  # Disable wandb in debug mode
        else:
            wandb.init(
                project=args.project_name,
                name=args.exp_name,
                config=vars(args),
                dir=args.log_dir,
            )
    else:
        writer = None
        logger = get_root_logger(name='train')  # For other ranks
        wandb.init(mode="disabled")  # Disable wandb for non-master ranks
    device = torch.device(f'cuda:{args.local_rank}')
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Create model
    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    tokenizer = processor.tokenizer
    num_added_tokens = tokenizer.add_tokens("[CONT]")
    args.cont_token_idx = tokenizer("[CONT]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model_args = {
        "model": args.version,
        "vlm_out_dim": args.vlm_out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "cont_token_idx": args.cont_token_idx,
        "torch_dtype": torch_dtype,
        "attention": args.attention,
        "point_embd_dim": 512,
        "point_N_p": 64,
    }

    config = transformers.AutoConfig.from_pretrained(args.version)
    # vocab size: 151936 tokenizer size: 151666
    model = Hammer(config, **model_args).to(device)
    if args.local_rank == 0:
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} total parameters")

    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id

    model.vlm.enable_input_require_grads()
    for p in model.vlm.visual.parameters():
        p.requires_grad = False

    if args.lora_r > 0:
        # LoRA target modules [q_proj, v_proj]
        lora_target_modules = args.lora_target_modules.split(",")
        if args.local_rank == 0:
            logger.info(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_config)

    model.vlm.resize_token_embeddings(len(tokenizer))
    model.vlm.config.vocab_size = len(tokenizer)
    
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]):
            if args.local_rank == 0: 
                logger.info(f"Parameter {n} (shape {p.shape}) is trainable.")
            p.requires_grad = not (args.ce_loss_weight == 0.0)
        if any(
            [
                x in n
                for x in [
                    "projection", "point_backbone", "lift_to_3d", "afford_decoder"
                ]
            ]
        ):
            if args.local_rank == 0: 
                logger.info(f"Parameter {n} (shape {p.shape}) is trainable.")
            p.requires_grad = True
    for n, p in model.vlm.lm_head.named_parameters():
        if args.local_rank == 0:
            logger.info(f"Parameter model.vlm.lm_head.{n} (shape {p.shape}) is trainable.")
        p.requires_grad = True
    model.vlm.print_trainable_parameters()

    if args.dataset == "PIADv1":
        DatasetClass = PIADv1Afford
    elif args.dataset == "PIADv2":
        DatasetClass = PIADAfford
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset = DatasetClass(
        data_root=args.dataset_dir,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        split="train",
        setting=args.setting,
        model_name="qwen_vl",
        question_type=args.question_type,
    )

    val_dataset = DatasetClass(
        datasets="piad",
        data_root=args.dataset_dir,
        split="test",
        setting=args.setting,
        model_name="qwen_vl",
        question_type=args.question_type,
    )
    if args.local_rank == 0:
        logger.info(f"Training with {len(train_dataset)} examples.")
        logger.info(f"Validation with {len(val_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        # Add loss scaling for better numerical stability
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            model_name="qwen_vl",
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        if args.local_rank == 0:
            logger.info(f"Loading checkpoint from {args.resume}")
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        if args.local_rank == 0:
            logger.info(
                "Resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

    # Create distributed validation loader
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=args.local_rank,
            shuffle=False,
            drop_last=False
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            model_name="qwen_vl",
        ),
    )
    
    train_iter = iter(train_loader)
    best_score = 0.0
    best_model = None

    if args.local_rank == 0:
        logger.info("=" * 50)
        logger.info("TRAINING STARTED")
        logger.info("=" * 50)
        logger.info(f"Total epochs: {args.epochs}")
        logger.info(f"Steps per epoch: {args.steps_per_epoch}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
        logger.info(f"Effective batch size: {args.batch_size * args.grad_accumulation_steps * world_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Precision: {args.precision}")
        logger.info("=" * 50)

    for epoch in range(args.start_epoch, args.epochs):
        if args.local_rank == 0:
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        train_iter, steps = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args, logger)

        iou = validate(val_loader, model_engine, epoch, writer, args, logger, steps)
        is_best = iou > best_score
        best_score = max(best_score, iou)

        if is_best:
            if args.local_rank == 0:
                logger.info(f"New best IoU: {iou:.4f}, saving model...")

                if best_model and os.path.exists(best_model):
                    os.remove(best_model)
                best_model = os.path.join(args.log_dir, f"meta_log_iou_{best_score:.3f}.pth")
                    
            save_dir = os.path.join(args.log_dir, "model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch, "best_score": best_score},
                    best_model,
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                logger.info(f"Checkpoint saved to {save_dir}")

    if args.local_rank == 0:
        wandb.finish()
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 50)


if __name__ == "__main__":
    main(sys.argv[1:])

