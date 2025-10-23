import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import time
import tqdm
import glob
import argparse
from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from configs import hf_token, prompt_formats, llm_domains
import torch.nn.functional as F

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_generator.data_loader import DataCreator
from data_generator.data_helper import construct_open_ended_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, AutoTokenizer
from model_helper import load_image
from datasets import load_dataset


def load_model(model_path):
    if "llava" in model_path:
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            token=hf_token
        ).to("cuda")
        processor = LlavaNextProcessor.from_pretrained(model_id)
    elif "Qwen" in model_path:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    elif "InternVL2" in model_path:
        model_id = "OpenGVLab/InternVL2-8B"
        model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    return model, processor

def hook_models(model, model_name):
    # --- Containers for hook outputs ---
    visual_features = {}

    # --- Register hooks ---
    if "llava" in model_name:
        def hook_visual(module, input, output):
            visual_features["raw"] = output.last_hidden_state.detach().cpu()
        h1 = model.vision_tower.register_forward_hook(hook_visual)

    elif "Qwen" in model_name:
        def hook_visual(module, input, output):
            visual_features["raw"] = output.detach().cpu()    
        h1 = model.visual.register_forward_hook(hook_visual)

    elif "InternVL2" in model_name:
        def hook_visual(module, input, output):
            visual_features["raw"] = output.last_hidden_state.detach().cpu()
        h1 = model.vision_model.register_forward_hook(hook_visual)

    return visual_features
    

def feed_images(image, model, processor, model_name):
    output = None
    question = "What is in this image?"
    # question = "What is in this image?\nAnswer the question shortly."
    if ("llava" in model_name) or ("Qwen" in model_name):
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]

        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt"
        ).to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=50)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True)

    elif "InternVL2" in model_name:
        # Load image
        generation_config = dict(max_new_tokens=100, return_dict_in_generate=False, output_scores=True)
        pixel_values = load_image(image).to(torch.bfloat16).cuda()
        output = model.chat(processor, pixel_values, question, generation_config)
        
    return output
    


def run(args):
    print(args)

    ds = load_dataset("HuggingFaceM4/A-OKVQA", token=hf_token)
    model, processor = load_model(args.model_name)

    visual_features = hook_models(model, args.model_name)
    
    save_dir = os.path.join(parent_dir, "results", "visual_features", "okvqa", args.dataset_type)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # else:
    #     save_path = os.path.join(save_dir, f"{args.model_name}_vis_embed_{2000}.pt")
    #     embeddings = torch.load(save_path)
    embeddings = []
    for i, example in enumerate(tqdm.tqdm(ds[args.dataset_type])):
        im = example["image"]
        output = feed_images(im, model, processor, args.model_name)
        embeddings.append(visual_features["raw"])
        
        if i % args.checkpoint_count == 0 and i > 0:
            save_path = os.path.join(save_dir, f"{args.model_name}_vis_embed_{i}.pt")
            torch.save(embeddings, save_path)

        # tqdm.tqdm.write(f"{output[:100]}... {visual_features['raw'].shape}")

        if i == args.num_samples:
            print("Loop finished")
            break

    save_path = os.path.join(save_dir, f"{args.model_name}_vis_embed.pt")
    torch.save(embeddings, save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="ocr", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--model_name", type=str, default="InternVL2-8B",
                        choices=["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "Qwen2.5-VL-7B-Instruct", "InternVL2-8B"])
    parser.add_argument("--dataset_type", type= str, default="train", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--checkpoint_count", type=int, default=1500)
    arguments = parser.parse_args()
    run(arguments)
