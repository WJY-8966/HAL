import torch
import torch.nn.functional as F
from utils.utils import seed_torch
import random
from collections import defaultdict


def normalize_embeddings(embeddings):
    """
    将 embedding L2 归一化到单位球面
    """
    return F.normalize(embeddings, dim=1)

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
    x: [N, D], r: [L, K, D] → L 个表，每个表 K 个投影向量
    返回: [L, N, K] binary hash bits
    """
    L, K, D = r.shape
    N = x.shape[0]
    proj = torch.einsum('nd,lkd->lnk', x, r)  # [L, N, K]
    bits = (proj > 0).int()
    return bits  # [L, N, K]

def lsh_bucket_index(bits):
    """
    bits: [L, N, K]
    返回: [L, N] → 每张表中每个样本的哈希桶编号
    """
    powers = 2 ** torch.arange(bits.shape[-1], device=bits.device)
    return (bits * powers).sum(dim=-1)  # [L, N]

def build_lsh_tables(bits_idx):
    """
    bits_idx: [L, N] → 每个样本在每张表的桶编号
    返回: L 个哈希表，每个表是 bucket_id → list of indices
    """
    L, N = bits_idx.shape
    tables = []
    for l in range(L):
        table = defaultdict(list)
        for i in range(N):
            bucket = int(bits_idx[l, i].item())
            table[bucket].append(i)
        tables.append(table)
    return tables  # List[Dict[int, List[int]]]

def query_candidates(query_bits, tables, db_bits_idx, topk=50):
    """
    query_bits: [L, K], query 的 L 张哈希码
    tables: List[Dict[int, List[int]]]，哈希表
    返回：候选集合（indices），可能含重复
    """
    L = len(tables)
    bucket_ids = (query_bits * (2 ** torch.arange(query_bits.shape[-1], device=query_bits.device))).sum(dim=1)  # [L]
    candidates = []
    for l in range(L):
        b = int(bucket_ids[l].item())
        candidates.extend(tables[l].get(b, []))
    return list(set(candidates))  # 去重返回

def evaluate_multitable_lsh_recall(img_embeddings, txt_embeddings, L=10, K=16, topk=(1,5,10), max_candidates=100):
    """
    img_embeddings, txt_embeddings: [N, D] 已归一化
    多哈希表 LSH 检索评估 Recall
    """
    device = img_embeddings.device
    N, D = img_embeddings.shape

    # r = generate_orthogonal_lsh_vectors(shift, dim=D, num_vecs=K * L).view(L, K, D)

    # Step 1: 构建随机投影向量
    r = F.normalize(torch.randn(L, K, D, device=device), dim=-1)  # [L, K, D]

    # Step 2: 获取哈希码 + 桶编号
    img_bits = lsh_hash_bits(img_embeddings, r)   # [L, N, K]
    txt_bits = lsh_hash_bits(txt_embeddings, r)   # [L, N, K]
    img_idx = lsh_bucket_index(img_bits)          # [L, N]
    txt_idx = lsh_bucket_index(txt_bits)          # [L, N]

    # Step 3: 构建文本的 LSH 表（供图像检索）
    txt_tables = build_lsh_tables(txt_idx)
    img_tables = build_lsh_tables(img_idx)

    recall_metrics = {f'R@{k}_i2t': 0.0 for k in topk}
    recall_metrics.update({f'R@{k}_t2i': 0.0 for k in topk})

    # Step 4: image → text 检索
    for i in range(N):
        query_bits = img_bits[:, i, :]  # [L, K]
        candidates = query_candidates(query_bits, txt_tables, txt_idx, topk=max_candidates)
        if len(candidates) == 0:
            continue
        # 在 candidates 中计算 Hamming 距离
        dist = (txt_bits[:, candidates, :] != query_bits.unsqueeze(1)).sum(dim=(0, 2))  # [len(candidates)]
        topk_idx = torch.topk(-dist, k=min(max(topk), len(dist)))[1]  # 越小越相似
        top_indices = [candidates[j] for j in topk_idx.cpu()]
        for k in topk:
            if i in top_indices[:k]:
                recall_metrics[f'R@{k}_i2t'] += 1

    # Step 5: text → image 检索
    for i in range(N):
        query_bits = txt_bits[:, i, :]  # [L, K]
        candidates = query_candidates(query_bits, img_tables, img_idx, topk=max_candidates)
        if len(candidates) == 0:
            continue
        dist = (img_bits[:, candidates, :] != query_bits.unsqueeze(1)).sum(dim=(0, 2))  # [len(candidates)]
        topk_idx = torch.topk(-dist, k=min(max(topk), len(dist)))[1]
        top_indices = [candidates[j] for j in topk_idx.cpu()]
        for k in topk:
            if i in top_indices[:k]:
                recall_metrics[f'R@{k}_t2i'] += 1

    # Step 6: 平均
    for k in topk:
        recall_metrics[f'R@{k}_i2t'] /= N
        recall_metrics[f'R@{k}_t2i'] /= N

    return recall_metrics

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

if __name__ == '__main__':
    pretrained_file = '/data/zhouxiaokai/codes/FedML/pretrained_embeddings/Flicker30k/imagebind_embeddings.pt'
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
    # lsh_planes = generate_orthogonal_lsh_vectors_without_training(h, dim=image_embeddings.shape[1], num_vecs=512)
    lsh_planes = generate_lsh_vectors(dim=image_embeddings.shape[1], num_vecs=16)

    # 假设 image_embeddings 和 text_embeddings 已经 L2-normalized 并在 GPU 上
    metrics = evaluate_multitable_lsh_recall(
        image_embeddings, text_embeddings,
        L=64,  # 哈希表数量
        K=16,  # 每张表使用多少位
        topk=(1, 5, 10),
        max_candidates=100000  # 每次检索的最大候选
    )
    print(metrics)