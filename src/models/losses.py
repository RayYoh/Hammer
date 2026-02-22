import torch
import torch.nn.functional as F


def ca_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
):
    """
    Compute the Class Activation (CA) loss, which combines cross-entropy and DICE loss.
    Args:
        pred: A float tensor of arbitrary shape. The predictions for each example.
        target: A float tensor with the same shape as pred. Stores the binary
                classification label for each element in pred
                (0 for the negative class and 1 for the positive class).
        gamma: Focusing parameter for the Focal Loss component.
        alpha: Weighting factor for the positive class in the Focal Loss component.
    Returns:
        Loss tensor combining cross-entropy and DICE loss.
    """
    tmp1 = -(1 - alpha) * torch.mul(
        pred**gamma, torch.mul(1 - target, torch.log(1 - pred + 1e-6))
    )
    tmp2 = -alpha * torch.mul(
        (1 - pred)**gamma, torch.mul(target, torch.log(pred + 1e-6))
    )
    tmp = tmp1 + tmp2
    ce_loss = torch.sum(torch.mean(tmp, (0, 1)))

    intersection_positive = torch.sum(pred * target, 1)
    cardinality_positive = torch.sum(torch.abs(pred) + torch.abs(target), 1)
    dice_positive = (intersection_positive + 1e-6) / (cardinality_positive + 1e-6)
    intersection_negative = torch.sum((1.0 - pred) * (1.0 - target), 1)
    cardinality_negative = torch.sum(2 - torch.abs(pred) - torch.abs(target), 1)
    dice_negative = (intersection_negative + 1e-6) / (cardinality_negative + 1e-6)
    tmp3 = torch.mean(1.5 - dice_positive - dice_negative, 0)
    dice_loss = torch.sum(tmp3)
    return ce_loss + 1.0 * dice_loss


def dice_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    num_masks: float, 
    scale=1000, 
    eps=1e-6
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    num_masks: float
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def batch_dice_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor
):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor
):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw

