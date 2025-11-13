import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils import seed_torch
import random




def generate_lsh_vectors(dim, num_vecs):
    """
    生成 num_vecs 个 dim 维的随机投影向量（不考虑 h，只为兼容函数签名）

    Args:
        dim (int): 向量维度
        num_vecs (int): 生成投影向量数量

    Returns:
        torch.Tensor: shape [num_vecs, dim] 的单位向量集合
    """
    rand_vecs = torch.randn(num_vecs, dim).to('cuda:0')  # 生成随机向量
    rand_vecs = F.normalize(rand_vecs, dim=1)  # 每个向量 L2 归一化
    return rand_vecs

def lsh_hash_embedding(embeddings, lsh_vectors):
    """
    embeddings: [N, D]
    lsh_vectors: [K, D]
    return: [N, K] binary hash (0/1)
    """
    projections = embeddings @ lsh_vectors.T  # [N, K]
    projections = F.normalize(projections, dim=1)
    return (projections > 0).int()


def sign_hashing_binary(embeddings):
    """
    将实值 embedding 转换为 0/1 编码的 Hamming 向量
    Args:
        embeddings (torch.Tensor): [N, D] 实值向量
    Returns:
        torch.Tensor: [N, D] 0/1 二值向量
    """
    return (embeddings > 0).int()

def sign_hashing_binary_Gaussian(embeddings, out_dim=512):
    """
    将实值 embedding 转换为 0/1 编码的 Hamming 向量
    Args:
        embeddings (torch.Tensor): [N, D] 实值向量
    Returns:
        torch.Tensor: [N, D] 0/1 二值向量
    """
    N, D = embeddings.shape
    proj = torch.randn(D, out_dim, device=embeddings.device)
    projected = embeddings @ proj

    return (projected >= 0).int()

def hamming_distance(a, b):
    """
    a: [N, K]
    b: [M, K]
    return: [N, M] Hamming distance matrix
    """
    return (a.unsqueeze(1) != b.unsqueeze(0)).sum(dim=2)

def evaluate_hamming_recall_chunked(img_embeddings, txt_embeddings, recall_ks=[1, 5, 10], chunk_size=128):
    """
    分块计算 Hamming-based recall，避免爆显存。
    输入：
        img_embeddings: [N, D]
        txt_embeddings: [N, D]
        chunk_size: 每个 chunk 的大小
    """
    assert img_embeddings.shape[0] == txt_embeddings.shape[0]
    N = img_embeddings.shape[0]
    device = img_embeddings.device
    target = torch.arange(N, device=device)

    # 二值化
    img_bin = sign_hashing_binary(img_embeddings).to(device)  # [N, D]
    txt_bin = sign_hashing_binary(txt_embeddings).to(device)  # [N, D]

    recalls = {f'R@{k}_i2t': 0.0 for k in recall_ks}
    recalls.update({f'R@{k}_t2i': 0.0 for k in recall_ks})

    # image → text
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        img_chunk = img_bin[start:end]  # [B, D]
        dists = hamming_distance(img_chunk, txt_bin)  # [B, N]
        sorted_indices = dists.argsort(dim=1)  # [B, N]

        chunk_targets = target[start:end]  # [B]
        for k in recall_ks:
            match = (sorted_indices[:, :k] == chunk_targets[:, None]).any(dim=1).float().sum().item()
            recalls[f'R@{k}_i2t'] += match

    # text → image
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        txt_chunk = txt_bin[start:end]  # [B, D]
        dists = hamming_distance(txt_chunk, img_bin)  # [B, N]
        sorted_indices = dists.argsort(dim=1)  # [B, N]

        chunk_targets = target[start:end]
        for k in recall_ks:
            match = (sorted_indices[:, :k] == chunk_targets[:, None]).any(dim=1).float().sum().item()
            recalls[f'R@{k}_t2i'] += match

    # 平均
    for k in recall_ks:
        recalls[f'R@{k}_i2t'] /= N
        recalls[f'R@{k}_t2i'] /= N

    return recalls


if __name__ == "__main__":
    # 假设你已经有以下 embedding
    # img_embeddings: [N, D] 图像特征
    # txt_embeddings: [N, D] 文本特征
    pretrained_file = '/data/zhouxiaokai/codes/FedML/pretrained_embeddings/Flicker30k/imagebind_embeddings.pt'
    data = torch.load(pretrained_file)
    print(len(data))
    seed_torch()

    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_set, test_set = data[:split_idx], data[split_idx:]

    train_img_embeddings = torch.stack([d['image_embedding'] for d in train_set])
    train_txt_embeddings = torch.stack([d['text_embedding'] for d in train_set])

    img_embeddings = torch.stack([d['image_embedding'] for d in test_set])
    txt_embeddings = torch.stack([d['text_embedding'] for d in test_set])

    img_embeddings = F.normalize(img_embeddings, dim=1)
    txt_embeddings = F.normalize(txt_embeddings, dim=1)

    img_embeddings = img_embeddings.to('cuda:0')
    txt_embeddings = txt_embeddings.to('cuda:0')

    metrics = evaluate_hamming_recall_chunked(img_embeddings, txt_embeddings, recall_ks=[1, 5, 10])
    print(metrics)