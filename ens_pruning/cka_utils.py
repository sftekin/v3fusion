import os
import itertools

import numpy as np
import torch
import torch.nn.functional as F


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def calc_cka_matrix(pooled_embeddings):
    num_models = len(pooled_embeddings)
    cka_matrix = np.ones((num_models, num_models))
    for i in range(num_models):
        for j in range(i + 1, num_models):
            X = pooled_embeddings[i].to(torch.float32).cpu().numpy()
            Y = pooled_embeddings[j].to(torch.float32).cpu().numpy()
            cka_matrix[i, j] = linear_CKA(X, Y)
            cka_matrix[j, i] = cka_matrix[i, j]
    return cka_matrix


def calc_focal_cka(model_ids, cka_matrix):
    model_ids = list(model_ids)
    if len(model_ids) < 2:
        return 1.0
    avg_cka = 0.0
    count = 0
    for i, j in itertools.combinations(model_ids, 2):
        avg_cka += cka_matrix[i, j]
        count += 1
    return avg_cka / count


def pool_embedding(embed):
    if embed.dim() >= 3:
        embed = embed[0]
        embed = embed[1:]
    last_dim = embed.shape[-1]
    embed = embed.reshape(-1, last_dim).mean(dim=0, keepdim=True)
    return F.normalize(embed, p=2, dim=1)


def load_pooled_embeddings(model_names, dataset_name, ds_split, project_dir):
    data_dir = os.path.join(project_dir, "results", "visual_features", dataset_name, ds_split)
    pooled_embeddings = []
    for mn in model_names:
        save_path = os.path.join(data_dir, f"{mn}_vis_embed.pt")
        raw_embeddings = torch.load(save_path, map_location="cpu", weights_only=True)
        pooled = torch.cat([pool_embedding(e) for e in raw_embeddings], dim=0)
        pooled_embeddings.append(pooled)
        del raw_embeddings
    return pooled_embeddings
