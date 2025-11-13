import torch
import torch.nn as nn
import torch.nn.functional as F


# class WeightGate(nn.Module):
#     def __init__(self, embed_dim, hidden_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             # nn.BatchNorm1d(512),
#             nn.Linear(embed_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, text_embeddings):  # [B, K, D]
#         B, K, D = text_embeddings.shape
#         scores = self.mlp(text_embeddings)  # [B, K, 1]
#         weights = F.softmax(scores, dim=1)  # [B, K, 1]
#         return weights  # 权重和为1

class WeightGate(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 或 Softmax（后续自己归一化也可以）
        )

    def forward(self, sim_scores):
        """
        sim_scores: [B, K] → 每个值单独送入MLP
        返回: weights [B, K]
        """
        # B, K = sim_scores.shape
        # x = sim_scores.unsqueeze(-1)   # →
        weights = self.mlp(sim_scores)            # → [B, K, 1]
        return weights.squeeze(-1)       # → [B, K]

class MergeGate(nn.Module):
    def __init__(self, dim, kconv=5, dropout=0.1):
        super(MergeGate, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kconv, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, sorted_text_feats):
        """
        sorted_text_feats: [B, K, D] — already sorted and weighted if needed
        Returns: [B, D]
        """
        x = sorted_text_feats.unsqueeze(1)  # [B, 1, K, D]
        x = self.conv(x)                    # [B, 1, 1, D] if K == kconv
        x = x.squeeze(2).squeeze(1)         # [B, D]
        x = self.dropout(x)
        x = self.mlp(x)                     # [B, D]
        return x

def compute_similarity(image_embed, text_embeds):
    """
    计算图像与 top-k 文本之间的相似度（默认使用余弦相似度）

    参数:
        image_embed: [B, D]            # 图像 embedding
        text_embeds: [B, K, D]         # 每个样本对应 K 个文本的 embedding

    返回:
        sim_scores: [B, K]             # 每个图像与其 K 个文本的相似度分数
    """
    # L2归一化
    image_embed_norm = F.normalize(image_embed, dim=-1)       # [B, D]
    text_embeds_norm = F.normalize(text_embeds, dim=-1)       # [B, K, D]

    # 扩展维度以进行批量点积
    image_embed_expanded = image_embed_norm.unsqueeze(1)      # [B, 1, D]

    # 点积计算余弦相似度
    sim_scores = torch.sum(image_embed_expanded * text_embeds_norm, dim=-1)  # [B, K]

    return sim_scores

def merge_gate_aggregate(weighted_texts, method='mean'):
    """
    将 [B, K, D] 的 weighted_texts 聚合为 [B, D]

    参数:
        weighted_texts: torch.Tensor，形状为 [B, K, D]
        method: str，'mean' 或 'sum'

    返回:
        torch.Tensor，形状为 [B, D]
    """
    assert method in ['mean', 'sum'], "method must be 'mean' or 'sum'"
    if method == 'mean':
        return weighted_texts.mean(dim=1)
    else:  # method == 'sum'
        return weighted_texts.sum(dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def l2norm(self, X):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X

    def forward(self, x):
        return self.l2norm(self.net(x))
        # return F.normalize(self.net(x), dim=1)

class FedSimModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, kconv):
        super().__init__()
        self.weight_gate = WeightGate()
        self.merge_gate = MergeGate(embed_dim, kconv)
        self.projection_head = ProjectionHead(embed_dim,embed_dim,embed_dim)  # e.g., a linear layer or MLP

    def forward(self, image_embedding, text_embeddings):
        """
        image_embedding: [B, D]
        text_embeddings: [B, K, D] (top-k text for each image)
        """
        # 1. compute similarity
        sim_scores = compute_similarity(image_embedding, text_embeddings)  # [B, K]

        # 2. weight gate: apply to each similarity scalar
        weight_inputs = sim_scores.unsqueeze(-1)  # [B, K, 1]
        weights = self.weight_gate(weight_inputs).squeeze(-1)  # [B, K]

        # 3. weighted text embeddings
        weighted_texts = text_embeddings * weights.unsqueeze(-1)  # [B, K, D]
        sort_indices = torch.argsort(weights, dim=1, descending=True)  # [B, K]
        sorted_texts = torch.gather(weighted_texts, 1, sort_indices.unsqueeze(-1).expand(-1, -1, text_embeddings.size(-1)))  # [B, K, D]


        # Merge
        fused_text = self.merge_gate(sorted_texts)  # [B, D]
        # fused_text = merge_gate_aggregate(sorted_texts, method='mean')  # [B, D] if using mean aggregation


        # Projection
        image_proj = self.projection_head(image_embedding)  # [B, D]
        text_proj = self.projection_head(fused_text)        # [B, D]

        return image_proj, text_proj