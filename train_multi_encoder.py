import os
import argparse
import tqdm

from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.multi_encoder import MultiEncoderSharedAutoencoder
from transformers import get_scheduler
import matplotlib.pyplot as plt


class EmbeddingLoader(Dataset):
    def __init__(self, dataset_name, model_names, split_type="train"):
        self.data_dir = os.path.join("results", "visual_features",
                                     dataset_name, split_type)
        print("loading embeddings")
        self.model_names = model_names
        self.embeddings = []
        for mn in model_names:
            save_path = os.path.join(self.data_dir, f"{mn}_vis_embed.pt")
            self.embeddings.append(torch.load(save_path, weights_only=True))
        print("embeddings are loaded")
        
        self.total_len = len(self.embeddings[0])
        
        # (Optional) sanity check: all model embeddings have same length
        assert all(len(e) == self.total_len for e in self.embeddings), \
            "All embeddings must have same number of samples."


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Return a list (or tuple) of embeddings for this sample
        return [emb[idx] for emb in self.embeddings]

    # def get(self, split_type="train"):
    #     for i in range(self.total_len):
    #         yield [self.embeddings[j][i] for j in range(len(self.model_names))]

def collate_fn(batch):
    # batch: list of samples, each is list of embeddings [emb_model1, emb_model2, ...]
    # We want to return a list of tensors, one per model, stacked across batch
    num_models = len(batch[0])
    batch_size = len(batch)
    batch_out = []
    for m in range(num_models):
        model_out = []
        for n in range(batch_size):
            embed = batch[n][m]
            if len(embed.shape) >= 3:
                embed = embed[0]
                embed = embed[1:]
            last_dim = embed.shape[-1]
            embed = embed.reshape(-1, last_dim).mean(dim=0, keepdims=True)
            pooled = F.normalize(embed, p=2, dim=1)
            model_out.append(pooled)
        model_out = torch.cat(model_out)
        batch_out.append(model_out)

    return batch_out


def loss_fn(latents, recons, embeddings, lambda_align=1.0, lambda_recon=1.0):
    # alignment: pairwise MSE among latents for same samples
    align_loss = 0.0
    M = len(latents)
    for i in range(M):
        for j in range(i+1, M):
            align_loss += F.mse_loss(latents[i], latents[j])
    # reconstruction
    recon_loss = sum(F.mse_loss(r, e) for r, e in zip(recons, embeddings))

    return lambda_align * align_loss + lambda_recon * recon_loss


def test_loop(enc_dec, val_loader, lambda_align=1.0, lambda_recon=1.0):
    enc_dec.eval()
    progress_bar = tqdm.tqdm(range(len(val_loader)))
    running_loss = []
    for batch in val_loader:
        batch = [k.to("cuda").float() for k in batch]
        with torch.no_grad():
            latents, recons = enc_dec(batch)
            loss = loss_fn(latents, recons, batch,
                           lambda_align=lambda_align, lambda_recon=lambda_recon)
        running_loss.append(loss.item())
        
        progress_bar.update(1)
        progress_bar.set_postfix({"Vall Loss": np.mean(running_loss)})
    mean_loss = np.mean(running_loss)
    return mean_loss


def run(args):
    print(args)
    model_names = ["llava-v1.6-vicuna-7b-hf", "Qwen2.5-VL-7B-Instruct", "InternVL2-8B"]
    dataset_name = "mmmu"
    
    train_dataset = EmbeddingLoader(dataset_name, model_names, split_type="validation")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    val_dataset = EmbeddingLoader(dataset_name, model_names, split_type="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    dims = [1024, 3584, 1024]
    enc_dec = MultiEncoderSharedAutoencoder(dims, latent_dim=512)
    
    progress_bar = tqdm.tqdm(range(len(train_dataloader) * args.num_epochs))
    optimizer = torch.optim.AdamW(enc_dec.parameters(), lr=5e-5)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

 
    enc_dec.train().to("cuda")
    best_dict = enc_dec.state_dict()
    best_val_score, tol = 99, 0
    running_train_loss = []
    running_val_loss = []
    for epoch in range(args.num_epochs):
        train_loss = []
        for batch in train_dataloader:
            # batch is a list: [batch_for_model1, batch_for_model2, ...]
            batch = [k.to("cuda").float() for k in batch]
            latents, recons = enc_dec(batch)
            loss = loss_fn(
                latents, recons, batch,
                lambda_align=args.lambda_align,
                lambda_recon=args.lambda_recon)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            train_loss.append(loss.item())
            progress_bar.set_postfix({"Train Loss": np.mean(train_loss)})
            torch.cuda.empty_cache()

        val_loss = test_loop(enc_dec, val_dataloader,
                             lambda_align=args.lambda_align, lambda_recon=args.lambda_recon)
        running_val_loss.append(val_loss)
        running_train_loss.append(np.mean(train_loss))
        if val_loss < best_val_score:
            best_val_score = val_loss
            best_dict = enc_dec.state_dict()
            tol = 0
        else:
            print(f"No improvement increasing tol: {tol + 1}...")
            tol += 1

        if tol >= 3:
            print("early stopping...")
            break

    print("Saving model...")
    model_save_path = os.path.join("results", "latent_map_models",
                                   f"latent_map_model.pth")
    torch.save(best_dict, model_save_path)
    
    # Convert to numpy arrays
    train_loss_array = np.array(running_train_loss)
    val_loss_array = np.array(running_val_loss)

    # Save as .npy files
    np.save("results/latent_map_models/train_loss.npy", train_loss_array)
    np.save("results/latent_map_models/val_loss.npy", val_loss_array)

    print(f"Train loss shape: {train_loss_array.shape}")
    print(f"Val loss shape: {val_loss_array.shape}")
    
    fig, ax = plt.subplots()
    x_axis = np.arange(len(train_loss_array))
    ax.plot(x_axis, train_loss_array, label="Train Loss")
    ax.plot(x_axis, val_loss_array, label="Validation Loss")
    ax.set_title('Loss Plot')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    plt.savefig("results/figures/loss.png", dpi=200, bbox_inches="tight")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="ocr", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lambda_align", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--dataset_type", type= str, default="train", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--checkpoint_count", type=int, default=500)
    arguments = parser.parse_args()
    run(arguments)
