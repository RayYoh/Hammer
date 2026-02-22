from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


OAFFORD_QUESTION_LIST = [
    "Segment the area on the {class_name} where the human is making direct contact in this image.",
    "Identify and mask the part of the {class_name} that the human is touching or interacting with in this scene.",
    "Show the contact points on the {class_name} where the human is physically connected to or interacting with it.",
    "Please provide a segmentation mask of the parts of the {class_name} that are in contact with the human.",
    "Highlight the areas on the {class_name} where there is physical interaction or contact with the human.",
]

OAFFORD_AFFORD_QUESTION_LIST = [
    "What type of affordance does the human-object interaction suggest? Then, segment the area on the {class_name} where the human is making contact.",
    "Describe the affordance provided by the interaction, and identify the part of the {class_name} that the human is touching or interacting with in this scene.",
    "Explain the affordance type shown by the contact points on the {class_name} where the human is physically connected. Then show the segmentation mask.",
    "Specify the affordance implied by the human's contact with the {class_name}, then provide a segmentation mask of the contact area.",
    "Describe the affordance associated with the physical interaction on the {class_name}, and highlight the contact areas with a segmentation mask.",
]

OAFFORD_ANSWER_LIST = [
    "It is [CONT].",
    "Sure, the object contact region is [CONT].",
    "Sure, the contact points on object is [CONT].",
    "Sure, the contact mask is [CONT].",
    "[CONT].",
]

OAFFORD_AFFORD_ANSWER_LIST = [
    "The affordance type is {affordance}, and the contact region is [CONT].",
    "This interaction suggests an affordance of {affordance}, and the object contact region is [CONT].",
    "The contact points indicate an affordance of {affordance}, with the mask at [CONT].",
    "This shows an affordance type of {affordance}, with contact at [CONT].",
    "Affordance: {affordance}, contact mask: [CONT].",
]

OAFFORD_AFFORD_OBJ_ANSWER_LIST = [
    "The affordance type is {affordance} with {class_name}, and the contact region is [CONT].",
    "This interaction suggests an affordance of {affordance} with {class_name}, and the object contact region is [CONT].",
    "The contact points indicate an affordance of {affordance} with {class_name}, with the mask at [CONT].",
    "This shows an affordance type of {affordance} with {class_name}, with contact at [CONT].",
    "Affordance: {affordance} with {class_name}, contact mask: [CONT].",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = "\t".join(entries)
        
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        message = " ".join(entries)
        
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
