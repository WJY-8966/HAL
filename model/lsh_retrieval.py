import torch
import torch.nn.functional as F
from utils.utils import seed_torch
import random

def normalize_embeddings(embeddings):
    """
    将 embedding L2 归一化到单位球面
    """
    return F.normalize(embeddings, dim=1)

import torch.nn as nn
from torch.utils.data import DataLoader

class ShiftDirectionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

def estimate_shift_direction(image_embeds, text_embeds, epochs=50, lr=1e-3):
    """
    拟合 shift 向量 h，表示模态之间的差异方向
    """
    device = image_embeds.device
    model = ShiftDirectionModel(image_embeds.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 构建训练数据：image - text → label 1（真实），text - image → label 0（负例）
    X = torch.cat([image_embeds - text_embeds, text_embeds - image_embeds], dim=0)
    y = torch.cat([torch.ones(len(image_embeds)), torch.zeros(len(image_embeds))]).unsqueeze(1).to(device)

    model.train()
    for _ in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 返回权重向量 h（单位化）
    h = model.linear.weight.data.squeeze()
    return F.normalize(h, dim=0)

def generate_orthogonal_lsh_vectors(h, dim, num_vecs):
    """
    在 h 的正交超平面上生成 LSH 投影向量
    """
    r = torch.randn(num_vecs, dim, device=h.device)
    h = h / h.norm()
    r_proj = r - (r @ h.unsqueeze(1)) * h.unsqueeze(0)
    r_proj = F.normalize(r_proj, dim=1)
    return r_proj  # [num_vecs, dim]

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


def estimate_shift_without_training(image_embeddings, text_embeddings):
    """
    估计 image 和 text embedding 之间的平均差异（shift 向量）
    """
    assert image_embeddings.shape == text_embeddings.shape
    shift = (image_embeddings - text_embeddings).mean(dim=0)
    shift = shift / shift.norm()  # 单位化
    return shift  # shape: [D]

def generate_orthogonal_lsh_vectors_without_training(shift, dim, num_vecs):
    """
    生成 num_vecs 个维度为 dim 的向量，与 shift 向量正交
    """
    shift = shift / shift.norm()
    basis = []

    for _ in range(num_vecs):
        r = torch.randn(dim).to('cuda:0')
        # 投影到 shift 的正交空间
        r_proj = r - (r @ shift) * shift
        r_proj = r_proj / r_proj.norm()
        basis.append(r_proj)

    return torch.stack(basis)  # shape: [num_vecs, dim]

def lsh_hash_bits(x, r):
    """
    x: [N, D], r: [K, D]
    返回: [N, K] binary {0, 1}
    """
    proj = x @ r.T
    bits = ((proj > 0).int())  # sign(x @ r): >0 → 1, ≤0 → 0
    return bits

def lsh_bucket_index(bits):
    """
    将二进制哈希码转换为整数桶编号
    """
    powers = 2 ** torch.arange(bits.shape[1], device=bits.device)
    return (bits * powers).sum(dim=1)

def compute_hamming_distance_matrix(A, B):
    """
    A: [N1, K] binary, B: [N2, K] binary
    返回 Hamming distance matrix: [N1, N2]
    """
    A = A.unsqueeze(1)  # [N1, 1, K]
    B = B.unsqueeze(0)  # [1, N2, K]
    return (A != B).sum(dim=2)  # [N1, N2]

def evaluate_hamming_recall(img_bits, txt_bits, topk=(1, 5, 10)):
    """
    输入：图像和文本的 LSH 二进制码
    返回：Recall@K
    """
    dist = compute_hamming_distance_matrix(img_bits, txt_bits)
    metrics = {}
    N = dist.shape[0]
    for k in topk:
        idx = dist.topk(k, largest=False, dim=1).indices  # [N, k]
        target = torch.arange(N, device=dist.device).unsqueeze(1)
        correct = (idx == target).any(dim=1).float().mean().item()
        metrics[f'R@{k}_i2t'] = correct
    return metrics

def compute_hamming_distance_chunked(A, B, chunk_size=512):
    """
    分块计算 Hamming 距离，避免爆显存
    A: [N1, K] binary, B: [N2, K] binary
    返回: [N1, N2] 距离矩阵
    """
    N1, K = A.shape
    N2 = B.shape[0]
    dist = []
    for start in range(0, N1, chunk_size):
        end = min(start + chunk_size, N1)
        chunk = A[start:end]  # [chunk, K]
        d = (chunk.unsqueeze(1) != B.unsqueeze(0)).sum(dim=2)  # [chunk, N2]
        dist.append(d)
    return torch.cat(dist, dim=0)  # [N1, N2]

def evaluate_hamming_recall_chunked(img_bits, txt_bits, topk=(1, 5, 10), chunk_size=512):
    """
    分块计算 recall，避免爆显存
    """
    N = img_bits.shape[0]
    metrics = {f'R@{k}_i2t': 0.0 for k in topk}
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dist = compute_hamming_distance_chunked(img_bits[start:end], txt_bits, chunk_size)
        for k in topk:
            idx = dist.topk(k, largest=False, dim=1).indices
            target = torch.arange(start, end, device=img_bits.device).unsqueeze(1)
            correct = (idx == target).any(dim=1).float().sum().item()
            metrics[f'R@{k}_i2t'] += correct
    for k in topk:
        metrics[f'R@{k}_i2t'] /= N
    return metrics




if __name__ == '__main__':
    pretrained_file = '/data2/kudret/codes/FedML/pretrained_embeddings/Flicker30k/imagebind_embeddings.pt'
    data = torch.load(pretrained_file)
    seed_torch()

    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_set, test_set = data[:split_idx], data[split_idx:]

    train_image_embeddings = torch.stack([d['image_embedding'] for d in train_set]).cuda()
    train_text_embeddings = torch.stack([d['text_embedding'] for d in train_set]).cuda()

    image_embeddings  = torch.stack([d['image_embedding'] for d in test_set])
    text_embeddings = torch.stack([d['text_embedding'] for d in test_set])

    # 假设已经有 image_embeddings, text_embeddings
    image_embeddings = normalize_embeddings(image_embeddings.cuda())
    text_embeddings = normalize_embeddings(text_embeddings.cuda())

    # Step 1: 拟合 shift 向量
    # h = estimate_shift_direction(train_image_embeddings, train_text_embeddings)
    h = estimate_shift_without_training(image_embeddings, text_embeddings)

    # Step 2: 构建正交 LSH 投影向量
    # lsh_planes = generate_orthogonal_lsh_vectors(h, dim=image_embeddings.shape[1], num_vecs=512)
    lsh_planes = generate_orthogonal_lsh_vectors_without_training(h, dim=image_embeddings.shape[1], num_vecs=512)
    # lsh_planes = generate_lsh_vectors(dim=image_embeddings.shape[1], num_vecs=1024)

    # Step 3: LSH 编码
    img_bits = lsh_hash_bits(image_embeddings, lsh_planes)
    txt_bits = lsh_hash_bits(text_embeddings, lsh_planes)

    # Step 4: 检索评估
    metrics = evaluate_hamming_recall_chunked(img_bits, txt_bits)
    print(metrics)
