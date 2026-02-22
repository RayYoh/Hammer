import os
import sys
import argparse
from functools import partial

import numpy as np

import torch
# Set TORCH_CUDA_ARCH_LIST before importing torch to avoid compilation warnings
# This should match your GPU architecture (8.9 for RTX 4090)
# More available in: https://developer.nvidia.com/cuda-gpus
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"

import transformers
from torch.utils.data import DataLoader

from src.models.hammer import Hammer
from src.utils.logger import get_root_logger
from src.utils.afford_dataset import PIADv1Afford, PIADAfford, collate_fn
from src.utils.utils import AverageMeter, dict_to_cuda, Summary
from tools.trainer import evaluate
from sklearn.metrics import roc_auc_score


def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    intersection = torch.minimum(map1, map2)
    return torch.sum(intersection)


def validate(val_loader, model, args):
    auc_meter = AverageMeter("AUC", ":6.3f", Summary.SUM)
    sim_meter = AverageMeter("SIM", ":6.3f", Summary.SUM)
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)
    mae_meter = AverageMeter("MAE", ":6.3f", Summary.SUM)

    torch_type = torch.float32
    if args.precision == "bf16":
        torch_type = torch.bfloat16
    elif args.precision == "fp16":
        torch_type = torch.float16

    model.eval()

    pred_affords, gt_affords = [], []
    for i, input_dict in enumerate(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        print(f"Evaluating {i+1}/{len(val_loader)}")
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch_type):
                output_dict = model(**input_dict)
            pred_affords.append(output_dict["pred_afford"].float())
            gt_affords.append(input_dict["gt_affords"].float())

    pred_affords = torch.cat(pred_affords, dim=0)
    gt_affords = torch.cat(gt_affords, dim=0)

    auc, iou, sim, mae = evaluate(pred_affords, gt_affords)
    
    auc_meter.update(auc)
    sim_meter.update(sim)
    iou_meter.update(iou)
    mae_meter.update(mae)

    print(
        f"AUC: {auc_meter.avg:.4f}, "
        f"SIM: {sim_meter.avg:.4f}, "
        f"IoU: {iou_meter.avg:.4f}, "
        f"MAE: {mae_meter.avg:.4f}"
    )
    return iou_meter.avg


def evaluate(pred, gt):
    B = gt.shape[0]
    iou_thres = np.linspace(0, 1, 20)
    sim_total, mae_total, auc_total, iou_total = 0, 0, 0, 0
    valid_samples = B

    for b in range(B):
        # Similarity and MAE
        sim = SIM(pred[b], gt[b])
        mae = torch.sum(torch.abs(pred[b] - gt[b])) / gt[b].shape[0]

        # Convert ground truth to binary mask
        gt_mask = (gt[b] >= 0.5).int()

        # Handle cases where all values are same (all 0s or all 1s)
        unique_gt = np.unique(gt_mask.cpu().numpy())
        if len(unique_gt) == 1:
            auc = float('nan')
            aiou = float('nan')
            valid_samples -= 1
        else:
            try:
                auc = roc_auc_score(gt_mask.cpu().numpy(), pred[b].cpu().numpy())
                temp_iou = []
                for thres in iou_thres:
                    pred_mask = (pred[b] > thres).int()
                    intersection = torch.sum(pred_mask & gt_mask)
                    union = torch.sum(pred_mask | gt_mask)
                    temp_iou.append(1. * intersection / union)
                temp_iou = torch.tensor(temp_iou)
                aiou = temp_iou.mean().item()
            except ValueError as e:
                print(f"ValueError for sample {b}: {e}")
                auc = float('nan')
                aiou = float('nan')
                valid_samples -= 1

        sim_total += sim.item()
        mae_total += mae.item()
        if not np.isnan(auc):
            auc_total += auc
        if not np.isnan(aiou):
            iou_total += aiou

    sim_avg = sim_total / B
    mae_avg = mae_total / B
    auc_avg = auc_total / max(valid_samples, 1)
    iou_avg = iou_total / max(valid_samples, 1)

    return auc_avg, iou_avg, sim_avg, mae_avg


def get_model(args):
    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=8192,
        padding_side="right",
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
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
        "vlm_out_dim": 256,
        "ce_loss_weight": 1.0,
        "mask_loss_weight": 2.0,
        "cont_token_idx": args.cont_token_idx,
        "torch_dtype": torch_dtype,
        "attention": "flash_attention_2",
        "point_embd_dim": 512,
        "point_N_p": 64,
    }
    config = transformers.AutoConfig.from_pretrained(args.version)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Hammer(config, **model_args).to(device)
    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id

    model.vlm.config.vocab_size = len(tokenizer)
    model.vlm.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model, processor, tokenizer, device


def parse_args(args):
    parser = argparse.ArgumentParser(description="Affordance Reasoning Model Evaluation")
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
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--log_base_dir", default="./exp", type=str)
    parser.add_argument(
        "--val_batch_size", default=4, type=int, help="validation batch size per device"
    )
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=2.0, type=float)
    parser.add_argument("--vlm_out_dim", default=256, type=int)
    parser.add_argument("--dataset", default="PIADv1", type=str, choices=["PIADv1", "PIADv2"])
    parser.add_argument("--question_type", default="afford_obj", type=str)
    parser.add_argument("--setting", default="Seen", type=str, choices=["Seen", "Unseen", "Unseen_afford", "Unseen_obj"])
    parser.add_argument("--weight", default="", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    model, processor, tokenizer, device = get_model(args)

    if args.dataset == "PIADv1":
        DatasetClass = PIADv1Afford
    elif args.dataset == "PIADv2":
        DatasetClass = PIADAfford
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    val_dataset = DatasetClass(
        datasets="piad",
        data_root=args.dataset_dir,
        split="test",
        setting=args.setting,
        model_name="qwen_vl",
        question_type=args.question_type,
    )

    # Create distributed validation loader
    val_sampler = None
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
    iou = validate(val_loader, model, args)


if __name__ == "__main__":
    main(sys.argv[1:])
