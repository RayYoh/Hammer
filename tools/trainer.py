import time
import wandb
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.utils import AverageMeter, ProgressMeter, Summary
from src.utils.utils import dict_to_cuda 


def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    intersection = torch.minimum(map1, map2)
    return torch.sum(intersection)


def train(train_loader, model, epoch, scheduler, writer, train_iter, args, logger):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    model.train()
    end = time.time()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16

    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            with torch.autocast(device_type='cuda', dtype=torch_dtype):
                output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_loss = output_dict["ca_loss"]

            losses.update(loss.item(), input_dict["points"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["points"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["points"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        steps = global_step + epoch * args.steps_per_epoch

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, steps)
                writer.add_scalar("train/ce_loss", ce_losses.avg, steps)
                writer.add_scalar("train/mask_loss", mask_losses.avg, steps)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, steps)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, steps)

                wandb.log({
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "metrics/total_secs_per_batch": batch_time.avg,
                    "metrics/data_secs_per_batch": data_time.avg,
                }, step=steps)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], steps)
                wandb.log({"train/lr": curr_lr[0]}, step=steps)
    return train_iter, steps


def validate(val_loader, model, epoch, writer, args, logger, steps):
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
        if i % 100 == 0 and args.local_rank == 0:
            logger.info(f"Evaluating batch {i}/{len(val_loader)}")
        input_dict = dict_to_cuda(input_dict)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch_type):
                output_dict = model(**input_dict)
            # pred_afford = torch.cat(output_dict["pred_afford"], dim=0)
            pred_affords.append(output_dict["pred_afford"].float())
            gt_affords.append(input_dict["gt_affords"].float())

    pred_affords = torch.cat(pred_affords, dim=0)
    gt_affords = torch.cat(gt_affords, dim=0)
    
    if args.distributed:
        world_size = torch.distributed.get_world_size()
        gathered_pred = [torch.zeros_like(pred_affords) for _ in range(world_size)]
        gathered_gt = [torch.zeros_like(gt_affords) for _ in range(world_size)]
        
        torch.distributed.all_gather(gathered_pred, pred_affords)
        torch.distributed.all_gather(gathered_gt, gt_affords)
        
        torch.distributed.barrier()
        
        pred_affords = torch.cat(gathered_pred, dim=0)
        gt_affords = torch.cat(gathered_gt, dim=0)

    auc, iou, sim, mae = evaluate(pred_affords, gt_affords)
    
    auc_meter.update(auc)
    sim_meter.update(sim)
    iou_meter.update(iou)
    mae_meter.update(mae)

    auc_meter.all_reduce()
    sim_meter.all_reduce()
    iou_meter.all_reduce()
    mae_meter.all_reduce()

    if args.local_rank == 0:
        writer.add_scalar("val/auc", auc_meter.avg, epoch)
        writer.add_scalar("val/sim", sim_meter.avg, epoch)
        writer.add_scalar("val/iou", iou_meter.avg, epoch)
        writer.add_scalar("val/mae", mae_meter.avg, epoch)

        wandb.log({
            "val/auc": auc_meter.avg,
            "val/sim": sim_meter.avg,
            "val/iou": iou_meter.avg,
            "val/mae": mae_meter.avg,
        }, step=steps)

        # Log the metrics
        logger.info(
            f"Epoch {epoch} - AUC: {auc_meter.avg:.4f}, "
            f"SIM: {sim_meter.avg:.4f}, IoU: {iou_meter.avg:.4f}, "
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

