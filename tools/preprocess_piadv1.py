"""
PIADv1 Point Affordance Extractor
==================================
Extracts per-point affordance annotations from the PIADv1 dataset and saves
them as .npy files grouped by (object category, affordance label).

Pipeline:
  1. Read the image list (Img_{split}.txt) and count occurrences of each
     (obj, affordance) pair.
  2. Read the point cloud list (Point_{split}.txt) and parse coordinates and
     affordance weights for each file.
  3. Keep only affordance labels that also appear in the image list
     (point-image alignment).
  4. Save each valid (obj, affordance) point cloud as an .npy array of shape
     (N, 4): xyz coordinates + single affordance weight column.
  5. Write all output file paths to Point_Extracted_{split}.txt.

Output structure:
  {data_root}/PIADv1/{setting}/Point_Extracted/{split}/{obj}/{affordance}/
      Point_{split}_{obj}_{affordance}_{num}.npy

Usage:
  python preprocess_piadv1.py --data_root /path/to/dataset --split train --setting Seen
  python preprocess_piadv1.py --data_root /path/to/dataset --split train --setting Unseen

Arguments:
  --data_root   Root directory of the dataset, must contain a PIADv1/ subdirectory (required)
  --split       Dataset split: train or test (default: train)
  --setting     Object setting: Seen or Unseen (default: Seen)
"""

import os
import argparse
import numpy as np


AFFORD_LABEL = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp',
    'pour', 'move', 'display', 'push', 'listen', 'wear', 'press', 'cut', 'stab'
]

SEEN_OBJ = [
    'Earphone', 'Bag', 'Chair', 'Refrigerator', 'Knife', 'Dishwasher',
    'Keyboard', 'Scissors', 'Table', 'StorageFurniture', 'Bottle', 'Bowl',
    'Microwave', 'Display', 'TrashCan', 'Hat', 'Clock', 'Door', 'Mug',
    'Faucet', 'Vase', 'Laptop', 'Bed'
]

UNSEEN_OBJ = [
    'Knife', 'Refrigerator', 'Earphone', 'Bag', 'Keyboard', 'Chair',
    'Hat', 'Door', 'TrashCan', 'Table', 'Faucet', 'StorageFurniture',
    'Bottle', 'Bowl', 'Display', 'Mug', 'Clock'
]


def read_point_file(data_root, setting, path):
    file_list = []
    with open(path, 'r') as f:
        base_path = os.path.join(data_root, 'PIADv1', setting)
        for file in f:
            file = file.strip()
            if "Data/" in file:
                file = os.path.join(base_path, *file.split('/')[2:])
            file_list.append(file)
    return file_list


def extract_point_file(point_file):
    coordinates = []
    with open(point_file, 'r') as f:
        for line in f:
            data = line.strip().split(' ')
            coordinates.append([float(x) for x in data[2:]])
    data_array = np.array(coordinates)
    coords = data_array[:, 0:3]
    afford = data_array[:, 3:]
    return coords, afford


def read_img_file(path, num_dict=None):
    file_list = []
    with open(path, 'r') as f:
        for file in f:
            file = file.strip()
            if num_dict is not None:
                obj, aff = file.split('/')[-3], file.split('/')[-2]
                num_dict[obj][aff] = num_dict[obj].get(aff, 0) + 1
            file_list.append(file)
    if num_dict is not None:
        return file_list, num_dict
    return file_list


def parse_args():
    parser = argparse.ArgumentParser(description="Extract point affordance data from PIADv1 dataset.")
    parser.add_argument(
        '--data_root', type=str, required=True,
        help='Root directory of the dataset (e.g., /path/to/data)'
    )
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'test'],
        help='Dataset split to process (default: train)'
    )
    parser.add_argument(
        '--setting', type=str, default='Seen', choices=['Seen', 'Unseen'],
        help='Object setting: Seen or Unseen (default: Seen)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    split = args.split
    setting = args.setting

    obj_list = SEEN_OBJ if setting == "Seen" else UNSEEN_OBJ
    num_dict = {item: {} for item in obj_list}

    img_path = os.path.join(data_root, f'PIADv1/{setting}/Img_{split}.txt')
    img_list, num_dict = read_img_file(img_path, num_dict)

    point_path = os.path.join(data_root, f'PIADv1/{setting}/Point_{split}.txt')
    point_list = read_point_file(data_root, setting, point_path)

    output_dir = os.path.join(data_root, f'PIADv1/{setting}/Point_Extracted/{split}')
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for i, point_file in enumerate(point_list):
        obj = point_file.split('/')[-2]
        num = point_file.split('/')[-1].split('_')[-1].split('.')[0]
        coords, afford = extract_point_file(point_file)

        non_zero_cols = np.any(afford != 0, axis=0)
        active_afford = np.where(non_zero_cols)[0]
        active_afford_labels = [AFFORD_LABEL[idx] for idx in active_afford]

        active_img_afford_labels = num_dict[obj].keys()
        valid_afford_labels = [label for label in active_afford_labels if label in active_img_afford_labels]
        print(f"Processing {obj} ({i+1}/{len(point_list)}) - Active affordances: {valid_afford_labels}")

        for label in valid_afford_labels:
            afford_idx = AFFORD_LABEL.index(label)
            data_array = np.hstack((coords, afford[:, afford_idx:afford_idx+1]))
            output_file = os.path.join(output_dir, obj, label, f'Point_{split}_{obj}_{label}_{num}.npy')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, data_array)
            output_file_path = os.path.join(
                f"Data/{setting}/Point_Extracted/{split}", obj, label,
                f'Point_{split}_{obj}_{label}_{num}.npy'
            )
            output_files.append(output_file_path + '\n')

    with open(os.path.join(data_root, f'PIADv1/{setting}/Point_Extracted_{split}.txt'), 'w') as f:
        f.writelines(output_files)

    print(f"Extracted point files saved to {output_dir}")
    print(f"Total files processed: {len(output_files)}")
    print("Done.")

