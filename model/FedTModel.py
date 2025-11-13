import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

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

class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_embs):
        # text_embs: (B, K, D) - K个文本嵌入
        # 在self-attention中，query, key, value都来自同一个输入
        query = self.query_proj(text_embs)  # (B, K, D)
        key = self.key_proj(text_embs)  # (B, K, D)
        value = self.value_proj(text_embs)  # (B, K, D)

        # 执行self-attention
        attn_output, _ = self.multihead_attn(query, key, value)  # (B, K, D)

        # 应用层归一化
        attn_output = self.norm(attn_output)  # (B, K, D)

        # 对K个文本嵌入取平均，得到单个融合嵌入
        fused = torch.mean(attn_output, dim=1)  #

        return fused

class FedT(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.fusion = SelfAttentionFusion(embed_dim, num_heads)
        # self.top_fusion = SelfAttentionFusion(embed_dim, num_heads)
        self.img_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

    def forward(self, image_emb, topk_text_emb):
        fused_txt = self.fusion(topk_text_emb)
        img_out = self.img_proj(image_emb)
        txt_out = self.txt_proj(fused_txt)
        return img_out, txt_out


