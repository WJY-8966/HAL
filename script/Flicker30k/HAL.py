"""
Orthogonal LSH + Private Hamming + Cross Attention-based fusion
"""
import torch
import random
import time
from data_loader.MSCOCODataset import EmbeddingDataset, embedding_collate_fn
from torch.utils.data import  DataLoader
from data_loader.utils import build_pseudo_aligned_dataset, build_pseudo_aligned_dataset_t2i
from utils.utils import seed_torch
from model.models import ClipLinear, ContrastiveLoss, PseudoAlignModel, T2IOnlyModel, I2TOnlyModel



def train_one_epoch(model, dataloader,  optimizer, criterion, device, mode='i2t'):
    model.train()
    total_loss = 0
    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)

        img_feat, txt_feat = model(img, txt, mode=mode)
        loss = criterion(img_feat, txt_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)




def evaluate_retrieval(model, dataloader, device='cuda', mode='i2t', t2i_only=False, i2t_only=False):
    model.eval()
    all_img, all_txt = [], []

    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)
        
        if t2i_only:
            # t2i-only 模型：直接传入 topk_img 和 text
            img_feat, txt_feat = model(img, txt)
        elif i2t_only:
            # i2t-only 模型：直接传入 img 和 topk_txt
            img_feat, txt_feat = model(img, txt)
        else:
            # 双向模型：需要指定 mode
            img_feat, txt_feat = model(img, txt, mode=mode)
        
        all_img.append(img_feat)
        all_txt.append(txt_feat)

    img_mat = torch.cat(all_img, dim=0)
    txt_mat = torch.cat(all_txt, dim=0)
    sim = img_mat @ txt_mat.T  # cosine similarity

    def recall_at_k(sim, k):
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        _, topk_i2t = sim.topk(k, dim=1)
        _, topk_t2i = sim.topk(k, dim=0)
        recall_i2t = (topk_i2t == targets[:, None]).any(dim=1).float().mean().item()
        recall_t2i = (topk_t2i == targets[None, :]).any(dim=0).float().mean().item()
        return recall_i2t, recall_t2i

    def mean_reciprocal_rank(sim, mode='i2t'):
        """
        MRR: Average of 1 / rank of ground-truth
        """
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        if mode == 'i2t':
            ranks = (sim.argsort(dim=1, descending=True) == targets[:, None]).nonzero()[:, 1]
        else:
            ranks = (sim.argsort(dim=0, descending=True) == targets[None, :]).nonzero()[:, 0]
        return (1.0 / (ranks + 1).float()).mean().item()

    def mean_average_precision(sim, mode='i2t'):
        """
        mAP: For 1 ground-truth per query
        """
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        ap = []
        if mode == 'i2t':
            sorted_indices = sim.argsort(dim=1, descending=True)  # [N, N]
            for i in range(N):
                correct = (sorted_indices[i] == i).nonzero(as_tuple=True)[0]
                if correct.numel() > 0:
                    rank = correct.item() + 1
                    ap.append(1.0 / rank)
                else:
                    ap.append(0.0)
        else:  # t2i
            sorted_indices = sim.argsort(dim=0, descending=True)  # [N, N]
            for i in range(N):
                correct = (sorted_indices[:, i] == i).nonzero(as_tuple=True)[0]
                if correct.numel() > 0:
                    rank = correct.item() + 1
                    ap.append(1.0 / rank)
                else:
                    ap.append(0.0)
        return sum(ap) / len(ap)

    metrics = {}
    for k in [1, 5, 10]:
        i2t, t2i = recall_at_k(sim, k)
        metrics[f'R@{k}_i2t'] = i2t
        metrics[f'R@{k}_t2i'] = t2i

    # MRR
    metrics['MRR_i2t'] = mean_reciprocal_rank(sim, 'i2t')
    metrics['MRR_t2i'] = mean_reciprocal_rank(sim, 't2i')

    # mAP
    metrics['mAP_i2t'] = mean_average_precision(sim, 'i2t')
    metrics['mAP_t2i'] = mean_average_precision(sim, 't2i')

    return metrics


def evaluate_retrieval_fused(model, dataloader_i2t, dataloader_t2i, device='cuda', alpha=0.5):
    """
    融合两个方向的信息进行一次性评估：
    - 先分别用 i2t 与 t2i 模式得到两个相似度矩阵
    - 然后做分数级融合：sim_fused = alpha * sim_i2t + (1 - alpha) * sim_t2i
    - 最终在 fused 相似度上计算检索指标（R@K / MRR / mAP）
    说明：这里的 i2t/t2i 指的是模型内部两个方向的 cross-attention，而非检索的行/列方向。
    """
    model.eval()
    all_img_i2t, all_txt_i2t = [], []
    all_img_t2i, all_txt_t2i = [], []

    # i2t: image 作为 query，text(top-k) 作为 key/value
    for batch in dataloader_i2t:
        img = batch['image_embedding'].to(device)         # [B, D]
        topk_txt = batch['text_embedding'].to(device)     # [B, K, D]
        img_feat, txt_feat = model(img, topk_txt, mode='i2t')
        all_img_i2t.append(img_feat)
        all_txt_i2t.append(txt_feat)

    # t2i: text 作为 query，image(top-k) 作为 key/value
    for batch in dataloader_t2i:
        topk_img = batch['image_embedding'].to(device)    # [B, K, D]
        txt = batch['text_embedding'].to(device)          # [B, D]
        img_feat, txt_feat = model(topk_img, txt, mode='t2i')
        all_img_t2i.append(img_feat)
        all_txt_t2i.append(txt_feat)

    img_i2t = torch.cat(all_img_i2t, dim=0)  # [N, D]
    txt_i2t = torch.cat(all_txt_i2t, dim=0)  # [N, D]
    img_t2i = torch.cat(all_img_t2i, dim=0)  # [N, D]
    txt_t2i = torch.cat(all_txt_t2i, dim=0)  # [N, D]

    # 两个方向各自的相似度矩阵
    sim_i2t = img_i2t @ txt_i2t.T   # [N, N]
    sim_t2i = img_t2i @ txt_t2i.T   # [N, N]

    # 融合：线性加权（可调 alpha）
    sim = alpha * sim_i2t + (1.0 - alpha) * sim_t2i

    def recall_at_k(sim, k):
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        _, topk_i2t = sim.topk(k, dim=1)   # 行：image -> text
        _, topk_t2i = sim.topk(k, dim=0)   # 列：text -> image
        recall_i2t = (topk_i2t == targets[:, None]).any(dim=1).float().mean().item()
        recall_t2i = (topk_t2i == targets[None, :]).any(dim=0).float().mean().item()
        return recall_i2t, recall_t2i

    def mean_reciprocal_rank(sim, mode='i2t'):
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        if mode == 'i2t':
            ranks = (sim.argsort(dim=1, descending=True) == targets[:, None]).nonzero()[:, 1]
        else:
            ranks = (sim.argsort(dim=0, descending=True) == targets[None, :]).nonzero()[:, 0]
        return (1.0 / (ranks + 1).float()).mean().item()

    def mean_average_precision(sim, mode='i2t'):
        N = sim.size(0)
        targets = torch.arange(N, device=sim.device)
        ap = []
        if mode == 'i2t':
            sorted_indices = sim.argsort(dim=1, descending=True)
            for i in range(N):
                correct = (sorted_indices[i] == i).nonzero(as_tuple=True)[0]
                if correct.numel() > 0:
                    rank = correct.item() + 1
                    ap.append(1.0 / rank)
                else:
                    ap.append(0.0)
        else:
            sorted_indices = sim.argsort(dim=0, descending=True)
            for i in range(N):
                correct = (sorted_indices[:, i] == i).nonzero(as_tuple=True)[0]
                if correct.numel() > 0:
                    rank = correct.item() + 1
                    ap.append(1.0 / rank)
                else:
                    ap.append(0.0)
        return sum(ap) / len(ap)

    metrics = {}
    for k in [1, 5, 10]:
        i2t, t2i = recall_at_k(sim, k)
        metrics[f'R@{k}_fused_i2t'] = i2t
        metrics[f'R@{k}_fused_t2i'] = t2i
    metrics['MRR_fused_i2t'] = mean_reciprocal_rank(sim, 'i2t')
    metrics['MRR_fused_t2i'] = mean_reciprocal_rank(sim, 't2i')
    metrics['mAP_fused_i2t'] = mean_average_precision(sim, 'i2t')
    metrics['mAP_fused_t2i'] = mean_average_precision(sim, 't2i')
    return metrics


def train_one_epoch_bidir(model, dataloader_i2t, dataloader_t2i, optimizer, criterion, device, t2i_weight=2.0):
    model.train()
    total_loss = 0.0
    num_steps = 0
    for (batch_i2t, batch_t2i) in zip(dataloader_i2t, dataloader_t2i):
        img_i2t = batch_i2t['image_embedding'].to(device)
        txt_i2t = batch_i2t['text_embedding'].to(device)

        img_t2i = batch_t2i['image_embedding'].to(device)
        txt_t2i = batch_t2i['text_embedding'].to(device)

        img_feat_i2t, txt_feat_i2t = model(img_i2t, txt_i2t, mode='i2t')
        img_feat_t2i, txt_feat_t2i = model(img_t2i, txt_t2i, mode='t2i')

        loss_i2t = criterion(img_feat_i2t, txt_feat_i2t)
        loss_t2i = criterion(img_feat_t2i, txt_feat_t2i)
        loss = loss_i2t + t2i_weight * loss_t2i

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

    return total_loss / max(num_steps, 1)


def train_one_epoch_t2i_only(model, dataloader_t2i, optimizer, criterion, device):
    """
    只训练 t2i 方向：text 作为 query，top-k images 作为 key/value
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    for batch_t2i in dataloader_t2i:
        # batch_t2i: 'image_embedding' 是 [B, K, D] (top-k images), 'text_embedding' 是 [B, D] (text)
        topk_img = batch_t2i['image_embedding'].to(device)  # [B, K, D]
        txt = batch_t2i['text_embedding'].to(device)         # [B, D]
        
        # 模型内部：text 作为 query，top-k images 作为 key/value
        img_feat, txt_feat = model(topk_img, txt)
        
        loss = criterion(img_feat, txt_feat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_steps += 1
    
    return total_loss / max(num_steps, 1)


def train_one_epoch_i2t_only(model, dataloader_i2t, optimizer, criterion, device):
    """
    只训练 i2t 方向：image 作为 query，top-k texts 作为 key/value
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    for batch_i2t in dataloader_i2t:
        # batch_i2t: 'image_embedding' 是 [B, D] (image), 'text_embedding' 是 [B, K, D] (top-k texts)
        img = batch_i2t['image_embedding'].to(device)      # [B, D]
        topk_txt = batch_i2t['text_embedding'].to(device)  # [B, K, D]
        
        # 模型内部：image 作为 query，top-k texts 作为 key/value
        img_feat, txt_feat = model(img, topk_txt)
        
        loss = criterion(img_feat, txt_feat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_steps += 1
    
    return total_loss / max(num_steps, 1)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 1e-4 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    pretrained_file = './pretrained_embeddings/Flicker30k/clip_embeddings.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 60
    batch_size = 512
    topk = 5
    data = torch.load(pretrained_file)
    print("data length:", len(data))
    seed_torch()
    for item in data:
        item['category'] = torch.tensor([-1], dtype=torch.long)  # Add dummy category
    random.shuffle(data)
    split_idx = int(0.8 * len(data))

    train_set, test_set = data[:split_idx], data[split_idx:]

    aligned_train_set = EmbeddingDataset(train_set, mode='aligned')
    aligned_test_set = EmbeddingDataset(test_set, mode='aligned')
    unaligned_train_set = EmbeddingDataset(train_set, mode='unaligned')
    unaligned_test_set = EmbeddingDataset(test_set, mode='unaligned')

    # construct pseudo aligned dataset
    construct_time = time.time()
    # Build i2t pseudo-aligned datasets (image -> top-k texts)
    pseudo_aligned_train_set_i2t = EmbeddingDataset(
        build_pseudo_aligned_dataset(unaligned_train_set, distance='PrivateHamming', topk=topk), mode='aligned')
    print('construct time (i2t):', time.time() - construct_time)
    construct_time = time.time()
    pseudo_aligned_test_set_i2t = EmbeddingDataset(
        build_pseudo_aligned_dataset(aligned_test_set, distance='PrivateHamming', topk=topk),  mode='aligned')

    # Build t2i pseudo-aligned datasets (text -> top-k images)
    pseudo_aligned_train_set_t2i = EmbeddingDataset(
        build_pseudo_aligned_dataset_t2i(unaligned_train_set, distance='PrivateHamming', topk=topk), mode='aligned')
    print('construct time (t2i):', time.time() - construct_time)
    construct_time = time.time()
    pseudo_aligned_test_set_t2i = EmbeddingDataset(
        build_pseudo_aligned_dataset_t2i(aligned_test_set, distance='PrivateHamming', topk=topk),  mode='aligned')

    # Dataloaders
    train_loader_i2t = DataLoader(pseudo_aligned_train_set_i2t, batch_size=batch_size, shuffle=True)
    test_loader_i2t = DataLoader(pseudo_aligned_test_set_i2t, batch_size=batch_size, shuffle=False)
    train_loader_t2i = DataLoader(pseudo_aligned_train_set_t2i, batch_size=batch_size, shuffle=True)
    test_loader_t2i = DataLoader(pseudo_aligned_test_set_t2i, batch_size=batch_size, shuffle=False)

    pseudo_aligned_model = PseudoAlignModel(embed_dim=512).to(device)

    pseudo_aligned_optimizer = torch.optim.Adam(pseudo_aligned_model.parameters(), lr=1e-4)
    pseudo_aligned_criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
    pseudo_aligned_criterion.to(device)

    # ========== 选择训练模式 ==========
    # 可选值: 'i2t_only', 't2i_only', 'bidir'
    TRAINING_MODE = 'bidir'
    # 训练日程：'bidir' 表示前半 i2t、后半 t2i；'i2t_full' 表示 60 个 epoch 全部 i2t
    TRAINING_SCHEDULE = 'bidir'  
    
    if TRAINING_MODE == 'i2t_only':
        print("=" * 60)
        print("Training mode: I2T-ONLY (image as query, top-k texts as key/value)")
        print("=" * 60)
        model = I2TOnlyModel(embed_dim=512).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
        criterion.to(device)
    elif TRAINING_MODE == 't2i_only':
        print("=" * 60)
        print("Training mode: T2I-ONLY (text as query, top-k images as key/value)")
        print("=" * 60)
        model = T2IOnlyModel(embed_dim=512).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
        criterion.to(device)
    else:  # bidir
        print("=" * 60)
        print("Training mode: BIDIRECTIONAL (both i2t and t2i)")
        print("=" * 60)
        model = pseudo_aligned_model
        optimizer = pseudo_aligned_optimizer
        criterion = pseudo_aligned_criterion
        t2i_weight = 2.0

    training_begin = time.time()

    for epoch in range(num_epochs):
        if TRAINING_SCHEDULE == 'i2t_full':
            # 全部 epoch 使用 i2t 方向
            train_loss = train_one_epoch(
                model, train_loader_i2t, optimizer, criterion, device, mode='i2t')
        else:
            if epoch < num_epochs // 2:
                # 只训练 i2t 方向
                train_loss = train_one_epoch(
                    model, train_loader_i2t, optimizer, criterion, device, mode='i2t')
            else:
                # 在方向切换的第一个 epoch，将优化器与 LR 重置到 1e-4
                if epoch == num_epochs // 2:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                # 只训练 t2i 方向
                train_loss = train_one_epoch(
                    model, train_loader_t2i, optimizer, criterion, device, mode='t2i')

        # 每个 epoch 仅评估融合指标
        metrics_fused = evaluate_retrieval_fused(model, test_loader_i2t, test_loader_t2i, device, alpha=0.5)

        phase = 'i2t-only' if (TRAINING_SCHEDULE == 'i2t_full' or epoch < num_epochs // 2) else 't2i-only'
        print(
            f"Epoch {epoch + 1} [{phase}] | Train Loss: {train_loss:.4f} | fused metrics: {metrics_fused}")

        # 阶段式 LR 衰减：每个阶段各自 0/10/20... 计数
        phase_epoch = epoch if epoch < num_epochs // 2 else (epoch - num_epochs // 2)
        lr = 1e-4 * (0.1 ** (phase_epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # for epoch in range(num_epochs):
    #     adjust_learning_rate(optimizer, epoch)

    #     if TRAINING_MODE == 'i2t_only':
    #         # 只训练 i2t 方向
    #         train_loss = train_one_epoch_i2t_only(
    #             model, train_loader_i2t, optimizer, criterion, device)
            
    #         # 评估 i2t 方向
    #         metrics_i2t = evaluate_retrieval(model, test_loader_i2t, device, mode='i2t', i2t_only=True)
    #         print(
    #             f"Epoch {epoch + 1} [i2t-only] | Train Loss: {train_loss:.4f} | i2t metrics: {metrics_i2t}")
    #     elif TRAINING_MODE == 't2i_only':
    #         # 只训练 t2i 方向
    #         train_loss = train_one_epoch_t2i_only(
    #             model, train_loader_t2i, optimizer, criterion, device)
            
    #         # 评估 t2i 方向
    #         metrics_t2i = evaluate_retrieval(model, test_loader_t2i, device, mode='t2i', t2i_only=True)
    #         print(
    #             f"Epoch {epoch + 1} [t2i-only] | Train Loss: {train_loss:.4f} | t2i metrics: {metrics_t2i}")
    #     else:  # bidir
    #         # 双向训练
    #         train_loss = train_one_epoch_bidir(
    #             model, train_loader_i2t, train_loader_t2i, optimizer, criterion, device, t2i_weight=t2i_weight)

    #         # 评估两个方向
    #         metrics_i2t = evaluate_retrieval(model, test_loader_i2t, device, mode='i2t', i2t_only=False, t2i_only=False)
    #         metrics_t2i = evaluate_retrieval(model, test_loader_t2i, device, mode='t2i', i2t_only=False, t2i_only=False)
    #         print(
    #             f"Epoch {epoch + 1} [bidir] | Train Loss: {train_loss:.4f} | i2t: {metrics_i2t} | t2i: {metrics_t2i}")


    print('training time:', time.time() - training_begin)


