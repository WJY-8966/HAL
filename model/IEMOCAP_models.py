import torch
from torch import nn
from torch.autograd import Variable

class ContrastiveLossTriModal(nn.Module):
    """
    三模态对比损失：video, text, audio
    """

    def __init__(self, margin=0, max_violation=False):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    def pairwise_loss(self, emb1, emb2):
        scores = self.sim(emb1, emb2)
        diagonal = scores.diag().view(emb1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(emb1.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def forward(self, video_emb, text_emb, audio_emb):
        loss_vt = self.pairwise_loss(video_emb, text_emb)
        loss_va = self.pairwise_loss(video_emb, audio_emb)
        loss_ta = self.pairwise_loss(text_emb, audio_emb)
        return loss_vt + loss_va + loss_ta

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_emb, topk_emb):
        # image_emb: (B, D), topk_text_emb: (B, K, D)
        query = self.query_proj(query_emb).unsqueeze(1)      # (B, 1, D)
        key = self.key_proj(topk_emb)                   # (B, K, D)
        value = self.value_proj(topk_emb)               # (B, K, D)

        fused, _ = self.multihead_attn(query, key, value)    # (B, 1, D)
        fused = self.norm(fused.squeeze(1))                  # (B, D)
        return fused

class PseudoAlignModel(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, num_classes=4):
        super().__init__()
        self.vt_fusion = CrossAttentionFusion(embed_dim, num_heads)
        self.va_fusion = CrossAttentionFusion(embed_dim, num_heads)
        self.video_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.audio_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim * 3, num_classes)


    def forward(self, video_emb, topk_text_emb, topk_audio_emb):
        fused_txt = self.vt_fusion(video_emb, topk_text_emb)
        fused_audio = self.va_fusion(video_emb, topk_audio_emb)
        video_out = self.video_proj(video_emb)
        txt_out = self.txt_proj(fused_txt)
        audio_out = self.audio_proj(fused_audio)
        fused = torch.cat([video_out, txt_out, audio_out], dim=-1)
        logits = self.classifier(fused)
        return logits

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(1024),
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

class MMFedAvgModel(nn.Module):

    def __init__(self, embed_dim=1024, num_classes=4):
        super(MMFedAvgModel, self).__init__()
        self.video_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.audio_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim * 3, num_classes)

    def forward(self, video_emb, text_emb, audio_emb):
        """One training step given images and captions.
        """
        video_out = self.video_proj(video_emb)
        txt_out = self.txt_proj(text_emb)
        audio_out = self.audio_proj(audio_emb)
        fused = torch.cat([video_out, txt_out, audio_out], dim=-1)
        logits = self.classifier(fused)

        return logits

