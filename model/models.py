import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class MultimodalClassifier(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=80, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_emb, txt_emb):
        x = torch.cat([img_emb, txt_emb], dim=1)
        return self.net(x)


# ========== Model ==========
class MultimodalRetrievalModel(nn.Module):
    def __init__(self, embed_dim, proj_dim=256):
        super(MultimodalRetrievalModel, self).__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, image_embedding, text_embedding):
        img_feat = self.image_proj(image_embedding)  # (B, proj_dim)
        txt_feat = self.text_proj(text_embedding)    # (B, proj_dim)

        # Normalize for cosine similarity
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        return img_feat, txt_feat



class ClipLinear(nn.Module):
    def __init__(self,  embed_dim=512):
        super(ClipLinear, self).__init__()


        # Add two FC for img and txt
        self.img_bn1 = nn.BatchNorm1d(512)
        self.img_fc1 = nn.Linear(512, embed_dim)
        self.img_bn2 = nn.BatchNorm1d(embed_dim)
        self.img_fc2 = nn.Linear(embed_dim, embed_dim)

        self.txt_bn1 = nn.BatchNorm1d(512)
        self.txt_fc1 = nn.Linear(512, embed_dim)
        self.txt_bn2 = nn.BatchNorm1d(embed_dim)
        self.txt_fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, images, captions, *args):
        """One training step given images and captions.
        """

        img_emb = F.gelu(self.img_fc1(self.img_bn1(images)))
        img_emb = self.img_fc2(self.img_bn2(img_emb))
        img_emb = self.l2norm(img_emb)


        cap_emb = F.gelu(self.txt_fc1(self.txt_bn1(captions)))
        cap_emb = self.txt_fc2(self.txt_bn2(cap_emb))
        cap_emb = self.l2norm(cap_emb)

        return img_emb, cap_emb

    def l2norm(self, X):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())
# Add Linear probing to CLIP as backbone

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

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_emb, topk_text_emb):
        query = self.query_proj(image_emb).unsqueeze(1)   # (B, 1, D)
        key = self.key_proj(topk_text_emb)                # (B, K, D)
        value = self.value_proj(topk_text_emb)            # (B, K, D)
        fused, _ = self.multihead_attn(query, key, value)
        fused = self.norm(fused.squeeze(1))               # (B, D)
        return fused



class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, a_emb, b_emb, mode='i2t'):
        """
        mode='i2t': a_emb=image [B, D], b_emb=topk_text [B, K, D]
        mode='t2i': a_emb=text  [B, D], b_emb=topk_img  [B, K, D]
        以 a_emb 作为 Query（单 token），以 b_emb 作为 Key/Value（K tokens）
        """
        if a_emb.dim() == 3:
            a_emb = a_emb.mean(dim=1)

        query = self.query_proj(a_emb).unsqueeze(1)    # (B, 1, D)
        key = self.key_proj(b_emb)                     # (B, K, D)
        value = self.value_proj(b_emb)                 # (B, K, D)
        fused, _ = self.multihead_attn(query, key, value)
        fused = self.norm(fused.squeeze(1))
        return fused


class PseudoAlignModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)
        # 为两个方向分别使用独立的 cross-attention 头（不共享参数）
        self.i2t_fusion = BidirectionalCrossAttentionFusion(embed_dim, num_heads)  # image -> text
        self.t2i_fusion = BidirectionalCrossAttentionFusion(embed_dim, num_heads)  # text -> image
        self.img_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

    def forward(self, image_emb, text_or_topk_emb, mode='i2t'):
        """
        mode='i2t':
            image_emb: [B, D]
            text_or_topk_emb: [B, K, D] (top-k text embeddings)
        mode='t2i':
            image_emb: [B, K, D] (top-k image embeddings)
            text_or_topk_emb: [B, D]
        输出均为 L2 归一化后的图/文投影，用于对比学习。
        """
        if mode == 'i2t':
            fused_txt = self.i2t_fusion(image_emb, text_or_topk_emb, mode='i2t')
            img_out = self.img_proj(image_emb)
            txt_out = self.txt_proj(fused_txt)
        elif mode == 't2i':
            # 将文本作为 a_emb，top-k 图像作为 b_emb
            fused_img = self.t2i_fusion(text_or_topk_emb, image_emb, mode='t2i')
            img_out = self.img_proj(fused_img)
            txt_out = self.txt_proj(text_or_topk_emb)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return img_out, txt_out


class T2IOnlyModel(nn.Module):
    """
    Text-to-Image only model: text as query, top-k images as key/value
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # Text-to-Image cross attention: text (query) -> top-k images (key/value)
        self.t2i_fusion = BidirectionalCrossAttentionFusion(embed_dim, num_heads)
        self.img_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

    def forward(self, topk_image_emb, text_emb):
        """
        Args:
            topk_image_emb: [B, K, D] - top-k image embeddings (作为 key/value)
            text_emb: [B, D] - text embedding (作为 query)
        Returns:
            img_out: [B, D] - 融合后的图像特征（L2 归一化）
            txt_out: [B, D] - 文本特征（L2 归一化）
        """
        # Text 作为 query，top-k images 作为 key/value
        # 融合后得到增强的图像语义表示
        fused_img = self.t2i_fusion(text_emb, topk_image_emb, mode='t2i')
        
        # 投影并归一化
        img_out = self.img_proj(fused_img)
        txt_out = self.txt_proj(text_emb)
        
        return img_out, txt_out


class I2TOnlyModel(nn.Module):
    """
    Image-to-Text only model: image as query, top-k texts as key/value
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # Image-to-Text cross attention: image (query) -> top-k texts (key/value)
        self.i2t_fusion = BidirectionalCrossAttentionFusion(embed_dim, num_heads)
        self.img_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)
        self.txt_proj = ProjectionHead(embed_dim, embed_dim, embed_dim)

    def forward(self, image_emb, topk_text_emb):
        """
        Args:
            image_emb: [B, D] - image embedding (作为 query)
            topk_text_emb: [B, K, D] - top-k text embeddings (作为 key/value)
        Returns:
            img_out: [B, D] - 图像特征（L2 归一化）
            txt_out: [B, D] - 融合后的文本特征（L2 归一化）
        """
        # Image 作为 query，top-k texts 作为 key/value
        # 融合后得到增强的文本语义表示
        fused_txt = self.i2t_fusion(image_emb, topk_text_emb, mode='i2t')
        
        # 投影并归一化
        img_out = self.img_proj(image_emb)
        txt_out = self.txt_proj(fused_txt)
        
        return img_out, txt_out


class Adapter(nn.Module):
    def __init__(self, embed_dim=512, reduction=16):
        super(Adapter, self).__init__()
        self.down = nn.Linear(embed_dim, embed_dim // reduction)
        self.activation = nn.ReLU()
        self.up = nn.Linear(embed_dim // reduction, embed_dim)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return residual + x  # Residual connection


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, D] → reshape to [B, 1, D] for attention
        x = x.unsqueeze(1)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)
        return x.squeeze(1)

class ClipEnhancedAdapterModel(nn.Module):
    def __init__(self, embed_dim=512, adapter_reduction=2, num_heads=8):
        super(ClipEnhancedAdapterModel, self).__init__()

        def make_branch():
            return nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Linear(512, embed_dim),
                nn.GELU(),
                nn.BatchNorm1d(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                Adapter(embed_dim, adapter_reduction),
                SelfAttention(embed_dim, num_heads)
            )

        self.image_branch = make_branch()
        self.text_branch = make_branch()

    def forward(self, image_emb, text_emb):
        img_feat = self.image_branch(image_emb)
        txt_feat = self.text_branch(text_emb)

        return self.l2norm(img_feat), self.l2norm(txt_feat)

    def l2norm(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)
        return x / norm




class MatchedContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=True, classification_weight=1.0, use_category=False, category_weight=0.1):
        super(MatchedContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.classification_weight = classification_weight
        self.use_category = use_category
        self.category_weight = category_weight
        self.sim = lambda x, y: F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)  # [B, B]
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_emb, txt_emb, is_matched=None, category=None):
        scores = self.sim(img_emb, txt_emb)  # similarity matrix: [B, B]
        diagonal = scores.diag().view(img_emb.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s.masked_fill_(mask, 0)
        cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        contrastive_loss = cost_s.sum() + cost_im.sum()

        # -------- Classification loss (optional, from is_matched) -------- #
        if is_matched is not None:
            matched_sim = F.cosine_similarity(img_emb, txt_emb)  # [B]
            class_loss = self.bce(matched_sim, is_matched.float())  # BCE loss
        else:
            class_loss = 0.0

        # -------- Category classification loss (optional) -------- #
        if self.use_category and category is not None:
            category_logits_img = F.linear(F.normalize(img_emb), F.normalize(img_emb))
            category_logits_txt = F.linear(F.normalize(txt_emb), F.normalize(txt_emb))
            category_loss = self.ce(category_logits_img, category) + self.ce(category_logits_txt, category)
        else:
            category_loss = 0.0

        # -------- Total loss -------- #
        total_loss = contrastive_loss + self.classification_weight * class_loss + self.category_weight * category_loss
        return total_loss

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        #if torch.cuda.is_available():
        I = I.to(im.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

