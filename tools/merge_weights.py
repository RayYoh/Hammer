import argparse
import os
import sys

import torch
import transformers
from peft import LoraConfig, get_peft_model

from src.models.hammer import Hammer
import glob
import os


def count_model_shards(model_path, pattern_prefix="pytorch_model-*-of-*.bin"):
    search_pattern = os.path.join(model_path, pattern_prefix)
    shard_files = glob.glob(search_pattern)
    shard_files.sort()
    
    if not shard_files:
        print(f"No files found in path {model_path} matching pattern '{pattern_prefix}'.")
        return 0, []
    
    total_shards = int(shard_files[-1].split("-")[-1].split(".")[0])
    return total_shards, shard_files


def add_base_layer_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "language_model" in key:
            if "q_proj.weight" in key or "q_proj.bias" in key:
                new_key = key.replace("q_proj", "q_proj.base_layer")
        new_state_dict[new_key] = value
    return new_state_dict


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into the base model and save the merged checkpoint."
    )
    parser.add_argument(
        "--version", default="Qwen/Qwen2.5-VL-3B-Instruct", type=str, help="pretrained model version"
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--model_max_length", default=8192, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--lora_target_modules", default="q_proj, v_proj", type=str)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=2.0, type=float)
    parser.add_argument("--vlm_out_dim", default=256, type=int)
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="exp/PIADv1_HAMMER", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # vocab size: 151936 tokenizer size: 151666
    model = Hammer(config, **model_args).to(device)
    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id

    if args.lora_r > 0:
        # LoRA target modules [q_proj, v_proj]
        lora_target_modules = args.lora_target_modules.split(",")
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

    print(f"Loading checkpoint from {args.weight} ...")
    state_dict = {}
    total_shards, shard_paths = count_model_shards(args.weight)
    print(f"Total shards found: {total_shards}")
    print("Shard files:", shard_paths)

    for path in shard_paths:
        state_dict_shard = torch.load(path, map_location='cpu', weights_only=True)
        state_dict.update(state_dict_shard)
        del state_dict_shard

    state_dict = add_base_layer_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    print("Missing Keys:", missing_keys)
    print("Unexpected Keys:", unexpected_keys)

    if hasattr(model.vlm, "merge_and_unload"):
        print("Merging LoRA weights into the base model...")
        model.vlm = model.vlm.merge_and_unload()
    else:
        raise AttributeError(
            "The model does not have merge_and_unload() method. "
            "Make sure you are using a PEFT version that supports merging LoRA adapters."
        )
    os.makedirs(args.save_path, exist_ok=True)
    saved_model = os.path.join(args.save_path, "model.bin")
    torch.save(model.state_dict(), saved_model)
    print(f"Merged model saved at {saved_model}")


if __name__ == "__main__":
    main(sys.argv[1:])
