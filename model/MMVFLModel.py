import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionExtractKV(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
        B, N, D = x.size()
        H = self.n_heads
        head_dim = D // H

        Q = self.q_proj(x).reshape(B, N, H, head_dim).transpose(1, 2)  # B x H x N x d
        K = self.k_proj(x).reshape(B, N, H, head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, H, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # B x H x N x N
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # B x H x N x d

        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(attn_output)
        return out, (K, V)

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

class CrossAttentionFromKV(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim * 2, dim)
        self.n_heads = n_heads
        self.dim = dim

    def forward(self, Q_input, KV_tuple):
        B, N, D = Q_input.size()
        H = self.n_heads
        head_dim = D // H

        Q = self.q_proj(Q_input).reshape(B, N, H, head_dim).transpose(1, 2)
        K, V = KV_tuple  # Both are B x H x N x d

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # B x H x N x d

        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        fused = torch.cat([Q_input, attn_output], dim=-1)  # concat along dim D
        fused = self.out_proj(fused)
        return fused

class ModalAttentionModel(nn.Module):
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.image_self_att = SelfAttentionExtractKV(dim, n_heads)
        self.text_self_att = SelfAttentionExtractKV(dim, n_heads)
        # self.cross_att_image = CrossAttentionFromKV(dim, n_heads)
        # self.cross_att_text = CrossAttentionFromKV(dim, n_heads)
        self.image_proj = ProjectionHead(input_dim=dim, hidden_dim=dim, output_dim=dim)
        self.text_proj = ProjectionHead(input_dim=dim, hidden_dim=dim, output_dim=dim)

    def forward(self, image_emb, text_emb):
        # Step 1: Self-attention for each modality
        image_feat, (image_K, image_V) = self.image_self_att(image_emb)
        text_feat, (text_K, text_V) = self.text_self_att(text_emb)

        # # Step 2: Use other's K/V for cross-modal attention
        # image_out = self.cross_att_image(image_feat, (text_K, text_V)).squeeze(1)  # image attends to text
        # text_out = self.cross_att_text(text_feat, (image_K, image_V)).squeeze(1) # text attends to image

        # Step 3: Project the outputs
        image_out = self.image_proj(image_feat.squeeze(1))
        text_out = self.text_proj(text_feat.squeeze(1))


        return image_out, text_out

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-Attention + Residual + Norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # MLP + Residual + Norm
        x = self.norm2(x + self.mlp(x))
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_emb, text_emb):
        """
        image_emb: (B, D)
        text_emb: (B, D)
        """
        B, D = image_emb.shape

        # Expand to (B, 1, D) to form sequence for attention
        query = self.query_proj(image_emb).unsqueeze(1)  # (B, 1, D)
        key = self.key_proj(text_emb).unsqueeze(1)       # (B, 1, D)
        value = self.value_proj(text_emb).unsqueeze(1)   # (B, 1, D)

        fused, _ = self.multihead_attn(query, key, value)  # (B, 1, D)
        fused = self.norm(fused.squeeze(1))                # (B, D)
        return fused


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class MMVFLModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, out_dim=512):
        super().__init__()
        # Self-Attention for image & text
        self.image_self = SelfAttentionBlock(embed_dim, num_heads)
        self.text_self = SelfAttentionBlock(embed_dim, num_heads)

        # Cross-Attention blocks
        self.cross_img2txt = CrossAttentionBlock(embed_dim, num_heads)
        self.cross_txt2img = CrossAttentionBlock(embed_dim, num_heads)

        # Projection heads
        self.img_proj = ProjectionHead(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim)
        self.txt_proj = ProjectionHead(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim)

    def forward(self, image_emb, text_emb):
        # Stage 1: self-attention
        image_feat = self.image_self(image_emb)  # [B, N, D]
        text_feat = self.text_self(text_emb)

        # Stage 2: cross-attention
        image_attn = self.cross_img2txt(image_feat, text_feat)  # image attends to text
        text_attn = self.cross_txt2img(text_feat, image_feat)  # text attends to image

        # print(image_attn.shape)

        # # Pooling (e.g., CLS token or average)
        # image_out = image_attn.mean(dim=1)  # [B, D]
        # text_out = text_attn.mean(dim=1)

        # Final projection
        image_vec = self.img_proj(image_attn)  # [B, out_dim]
        text_vec = self.txt_proj(text_attn)

        return image_vec, text_vec