import os
import copy
import random
import pickle
from torch.utils.data import Dataset

import numpy as np
import torch
from PIL import Image

from src.models.point_model.pointnet2 import pc_normalize

from .utils import OAFFORD_QUESTION_LIST, OAFFORD_ANSWER_LIST
from .utils import OAFFORD_AFFORD_QUESTION_LIST, OAFFORD_AFFORD_ANSWER_LIST
from .utils import OAFFORD_AFFORD_OBJ_ANSWER_LIST

from .conversation import get_default_conv_template


def collate_fn(
    batch, tokenizer=None, processor=None, model_name="qwen_vl"
):
    tokenizer = tokenizer or processor.tokenizer
    model_name = model_name.lower()

    image_list = []
    point_list = []
    afford_list = []
    conversation_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    obj_name_list = []
    afford_name_list = []

    for (
        image,
        point,
        afford,
        conversation,
        obj_name,
        afford_name,
        inference,
    ) in batch:
        conversation = [conversation]
        image_list.append(image)
        point_list.append(torch.from_numpy(point).float())
        afford_list.append(torch.from_numpy(afford).float())
        conversation_list.extend(conversation)
        cnt += len(conversation)
        offset_list.append(cnt)
        inferences.append(inference)
        obj_name_list.append(obj_name)
        afford_name_list.append(afford_name)

    points = torch.stack(point_list, dim=0)
    affords = torch.stack(afford_list, dim=0)
    if "qwen" in model_name:
        image_inputs = processor.image_processor(image_list, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        nb_token_list = []
        for i in range(len(image_grid_thw)):
            sample_image_grid_thw = image_grid_thw[i]
            nb_tokens = sample_image_grid_thw.prod()
            nb_token_list.append(nb_tokens)

        all_modified_conversations = []
        conversation_to_sample_idx = []
        duplicated_image_grid_thw = []
        for i in range(len(offset_list) - 1):
            start = offset_list[i]
            end = offset_list[i + 1]
            sample_conversations = conversation_list[start:end]

            sample_image_grid_thw = image_grid_thw[i]
            merge_length = processor.image_processor.merge_size ** 2
            nb_tokens = nb_token_list[i]
            num_placeholders = nb_tokens // merge_length

            for conv in sample_conversations:
                if processor.image_token in conv:
                    modified_conv = conv.replace(
                        processor.image_token,
                        "<|placeholder|>" * num_placeholders,
                        1
                    )
                    modified_conv = modified_conv.replace("<|placeholder|>", processor.image_token)
                else:
                    modified_conv = conv
                all_modified_conversations.append(modified_conv)
                conversation_to_sample_idx.append(i)

            num_conversations = offset_list[i + 1] - offset_list[i]
            duplicated_image_grid_thw.extend([sample_image_grid_thw] * num_conversations)

        text_inputs = tokenizer(all_modified_conversations, return_tensors="pt", padding=True)
        input_ids = text_inputs["input_ids"]
        attention_masks = text_inputs["attention_mask"]

        targets = input_ids.clone()
        targets[targets == tokenizer.pad_token_id] = -100
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_end_str = "<|im_end|>"

        assistant_start_tokens = tokenizer.encode(assistant_start_str, add_special_tokens=False)
        assistant_end_tokens = tokenizer.encode(assistant_end_str, add_special_tokens=False)

        for idx in range(targets.shape[0]):
            target = targets[idx]

            start_positions = []
            for pos in range(target.size(0) - len(assistant_start_tokens) + 1):
                if target[pos: pos + len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                    start_positions.append(pos + len(assistant_start_tokens))
            if not start_positions:
                raise ValueError(f"Assistant start tokens not found in sample {idx}")

            label = torch.full_like(target, -100)
            for i, start_idx in enumerate(start_positions):
                if i < len(start_positions) - 1:
                    search_end = start_positions[i + 1] - len(assistant_start_tokens)
                else:
                    search_end = target.size(0)
                found_end = False
                for pos in range(start_idx, search_end + 1):
                    if target[pos: pos + len(assistant_end_tokens)].tolist() == assistant_end_tokens:
                        end_idx = pos + 1
                        found_end = True
                        break
                if not found_end:
                    end_idx = search_end
                label[start_idx:end_idx] = target[start_idx:end_idx]
            targets[idx] = label

        cumsum_tokens = torch.cumsum(torch.tensor(nb_token_list), dim=0)
        duplicated_pixel_values = []
        for i in range(len(image_grid_thw)):
            start = cumsum_tokens[i - 1] if i > 0 else 0
            end = cumsum_tokens[i]
            image_segment = pixel_values[start:end]
            num_conversations = offset_list[i + 1] - offset_list[i]
            repeated_segment = image_segment.repeat(num_conversations, 1)
            duplicated_pixel_values.append(repeated_segment)
            
        pixel_values_batch = torch.cat(duplicated_pixel_values, dim=0)
        duplicated_image_grid_thw = torch.stack(duplicated_image_grid_thw)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported in this collate function.")
    
    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "pixel_values": pixel_values_batch,
        "image_grid_thw": duplicated_image_grid_thw,
        "points": points,
        "gt_affords": affords,
        "offset": torch.LongTensor(offset_list),
        "obj_names": obj_name_list,
        "afford_names": afford_name_list,
        "inference": inferences[0] if inferences else False,
    }


MAP = {
    'refrigerator': "Refrigerator", 
    'vase': "Vase",
    'door': "Door",
    'earphone': "Earphone",
    'clock': "Clock",
    'chair': "Chair",
    'laptop': "Laptop",
    'table': "Table",
    'dishwasher': "Dishwasher",
    'hat': "Hat",
    'bag': "Bag",
    'scissors': "Scissors",
    'keyboard': "Keyboard",
    'display': "Display",
    'bottle': "Bottle",
    'microwave': "Microwave",
    'trashcan': "TrashCan",
    'knife': "Knife",
    'bowl': "Bowl",
    'storagefurniture': "StorageFurniture",
    'bed': "Bed",
    'faucet': "Faucet",
    'mug': "Mug"
}


class GEAL(Dataset):
    def __init__(
        self,
        datasets="PIADv1",
        data_root="./data",
        model_name="qwen_vl",
        question_type='simple', # 'simple' or 'afford' or 'afford_obj',
        corrupt_type='scale',
        level=0,
        **kwargs
    ):
        super().__init__()

        file_name = f'{corrupt_type}_{level}.pkl'
        self.corrupt_type = corrupt_type
        self.level = level
        self.data_root = data_root
        self.datasets = datasets
        self.model_name = model_name.lower()
        self.question_type = question_type

        data_root = os.path.join(data_root, f"GEAL/{datasets}-C")
        img_path = os.path.join(data_root, 'image.txt')
        self.img_files = self.read_file(img_path)
        with open(os.path.join(data_root, 'point', file_name), 'rb') as f:
            self.anno = pickle.load(f)

        if self.question_type == 'simple':
            self.answer_list = OAFFORD_ANSWER_LIST
            self.question_list = OAFFORD_QUESTION_LIST
        elif self.question_type == 'afford':
            self.answer_list = OAFFORD_AFFORD_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST
        elif self.question_type == 'afford_obj':
            self.answer_list = OAFFORD_AFFORD_OBJ_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST

    def read_file(self, path):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            base_path = os.path.join(self.data_root, f'{self.datasets}/Seen')
            for file in files:
                file = file.strip('\n')
                if "Data/" in file:
                    file = os.path.join(base_path, *file.split('/')[2:])
                file_list.append(file)
            f.close()
        return file_list
    
    def generate_conversation(self, object_class, affordance):
        question_template = random.choice(self.question_list)
        question = question_template.format(class_name=object_class.lower())
        answer = random.choice(self.answer_list).format(
            affordance=affordance.lower(), class_name=object_class.lower()
        )
        conv = get_default_conv_template(self.model_name).copy()
        if "qwen" in self.model_name:
            user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{question}"
        else:
            user_message = f"<image>\n{question}"
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()
        return conversation
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        data = self.anno[idx]
        obj_name, afford_name = data['class'], data['affordance']
        obj_name = MAP[obj_name]
        if afford_name == 'wrap_grasp': afford_name = 'wrapgrasp'
        afford = np.expand_dims(data['mask'], axis=1)  # (N, 1)
        point = data['point']
        point = pc_normalize(point)
        point = point.transpose()

        img_file = self.img_files[idx]
        img_obj_name = img_file.split('/')[-3]
        img_afford_name = img_file.split('/')[-2]
        assert obj_name == img_obj_name and afford_name == img_afford_name, \
            f"{idx}, {obj_name} vs {img_obj_name}, {afford_name} vs {img_afford_name}"
        image_vlm = Image.open(img_file).convert('RGB')
        conversation = self.generate_conversation(obj_name, afford_name)
        inference = False

        return image_vlm, point, afford, conversation, obj_name, afford_name, inference

    
