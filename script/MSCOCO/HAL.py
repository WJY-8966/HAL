"""
Orthogonal LSH + Private Hamming + Cross Attention-based fusion
"""
import torch
import random
import time
from data_loader.MSCOCODataset import EmbeddingDataset, embedding_collate_fn
from torch.utils.data import DataLoader
from data_loader.utils import (
    build_pseudo_aligned_dataset,
    build_pseudo_aligned_dataset_t2i,
    build_bidirectional_pseudo_aligned_datasets_hashcoreset_reciprocal,
)
from utils.utils import seed_torch
from model.models import ClipLinear, ContrastiveLoss, RCMLoss, PseudoAlignModel, T2IOnlyModel, I2TOnlyModel


def train_one_epoch(model, dataloader, optimizer, criterion, device, mode='i2t'):
    model.train()
    total_loss = 0
    gate_means = []
    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)

        img_feat, txt_feat = model(img, txt, mode=mode)
        loss = criterion(img_feat, txt_feat)

        # 如需记录门控均值
        # gate_tensor = None
        # if mode == 't2i' and hasattr(model, 'img_gate') and getattr(model.img_gate, 'last_g', None) is not None:
        #     gate_tensor = model.img_gate.last_g
        # elif mode == 'i2t' and hasattr(model, 'txt_gate') and getattr(model.txt_gate, 'last_g', None) is not None:
        #     gate_tensor = model.txt_gate.last_g
        # if gate_tensor is not None:
        #     gate_means.append(gate_tensor.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if gate_means:
        print(f"[train_one_epoch][mode={mode}] gate mean over epoch: {sum(gate_means)/len(gate_means):.4f}")
    return total_loss / len(dataloader)


def evaluate_retrieval(model, dataloader, device='cuda', mode='i2t', t2i_only=False, i2t_only=False):
    """
    - 双向模型：通过 mode 指定 i2t 或 t2i 分支
    - 仅单向模型：通过 t2i_only / i2t_only 控制
    """
    model.eval()
    all_img, all_txt = [], []

    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)

        if t2i_only or i2t_only:
            img_feat, txt_feat = model(img, txt)
        else:
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


def evaluate_retrieval_fused(
    model,
    dataloader_i2t,
    dataloader_t2i,
    device='cuda',
    img_fuse_weight: float = 0.5,
    txt_fuse_weight: float = 0.5,
    l2_normalize_fused: bool = True,
):
    """
    融合两个方向的信息进行一次性评估：
    - 先分别用 i2t 与 t2i 模式得到两个相似度矩阵
    - 然后做特征级融合：将两分支的 img/txt 特征线性融合，再计算 sim_fused = img_fused @ txt_fused.T
    - 最终在 fused 相似度上计算检索指标（R@K / MRR / mAP）
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

    sim_i2t = img_i2t @ txt_i2t.T   # [N, N]，来自 i2t 分支
    sim_t2i = img_t2i @ txt_t2i.T   # [N, N]，来自 t2i 分支

    def _l2norm(x, eps: float = 1e-8):
        return x / (x.norm(dim=1, keepdim=True) + eps)

    # 特征级线性融合，再算相似度
    img_fused = img_fuse_weight * img_i2t + (1.0 - img_fuse_weight) * img_t2i
    txt_fused = txt_fuse_weight * txt_i2t + (1.0 - txt_fuse_weight) * txt_t2i
    if l2_normalize_fused:
        img_fused = _l2norm(img_fused)
        txt_fused = _l2norm(txt_fused)
    sim = img_fused @ txt_fused.T

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

    def fill_metrics_for_sim(sim_mat, prefix):
        """
        对给定相似度矩阵 sim_mat 计算检索指标，并用 prefix 作为前缀区分来源：
        - prefix='i2tBranch'：来自 i2t 分支的相似度矩阵
        - prefix='t2iBranch'：来自 t2i 分支的相似度矩阵
        - prefix='fused'    ：融合后的相似度矩阵
        """
        for k in [1, 5, 10]:
            i2t, t2i = recall_at_k(sim_mat, k)
            metrics[f'R@{k}_{prefix}_i2t'] = i2t
            metrics[f'R@{k}_{prefix}_t2i'] = t2i
        metrics[f'MRR_{prefix}_i2t'] = mean_reciprocal_rank(sim_mat, 'i2t')
        metrics[f'MRR_{prefix}_t2i'] = mean_reciprocal_rank(sim_mat, 't2i')
        metrics[f'mAP_{prefix}_i2t'] = mean_average_precision(sim_mat, 'i2t')
        metrics[f'mAP_{prefix}_t2i'] = mean_average_precision(sim_mat, 't2i')

    # 分别记录三个相似度矩阵的表现
    fill_metrics_for_sim(sim_i2t, 'i2tBranch')
    fill_metrics_for_sim(sim_t2i, 't2iBranch')
    fill_metrics_for_sim(sim, 'fused')

    return metrics


def train_one_epoch_bidir(model, dataloader_i2t, dataloader_t2i, optimizer, criterion, device, t2i_weight=2.0):
    """
    双向训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    gate_means_i2t = []
    gate_means_t2i = []
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

        if hasattr(model, 'txt_gate') and getattr(model.txt_gate, 'last_g', None) is not None:
            gate_means_i2t.append(model.txt_gate.last_g.mean().item())
        if hasattr(model, 'img_gate') and getattr(model.img_gate, 'last_g', None) is not None:
            gate_means_t2i.append(model.img_gate.last_g.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

    if gate_means_i2t:
        print(f"[train_one_epoch_bidir][i2t] gate mean over epoch: {sum(gate_means_i2t)/len(gate_means_i2t):.4f}")
    if gate_means_t2i:
        print(f"[train_one_epoch_bidir][t2i] gate mean over epoch: {sum(gate_means_t2i)/len(gate_means_t2i):.4f}")

    return total_loss / max(num_steps, 1)


def train_one_epoch_t2i_only(model, dataloader_t2i, optimizer, criterion, device):
    """
    只训练 t2i 方向：text 作为 query，top-k images 作为 key/value。
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    gate_means = []

    for batch_t2i in dataloader_t2i:
        topk_img = batch_t2i['image_embedding'].to(device)  # [B, K, D]
        txt = batch_t2i['text_embedding'].to(device)         # [B, D]

        img_feat, txt_feat = model(topk_img, txt)

        loss = criterion(img_feat, txt_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

        if hasattr(model, 'img_gate') and getattr(model.img_gate, 'last_g', None) is not None:
            gate_means.append(model.img_gate.last_g.mean().item())

    if gate_means:
        print(f"[train_one_epoch_t2i_only] gate mean over epoch: {sum(gate_means)/len(gate_means):.4f}")
    return total_loss / max(num_steps, 1)


def train_one_epoch_i2t_only(model, dataloader_i2t, optimizer, criterion, device):
    """
    只训练 i2t 方向：image 作为 query，top-k texts 作为 key/value。
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    gate_means = []

    for batch_i2t in dataloader_i2t:
        img = batch_i2t['image_embedding'].to(device)      # [B, D]
        topk_txt = batch_i2t['text_embedding'].to(device)  # [B, K, D]

        img_feat, txt_feat = model(img, topk_txt)

        loss = criterion(img_feat, txt_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

        if hasattr(model, 'txt_gate') and getattr(model.txt_gate, 'last_g', None) is not None:
            gate_means.append(model.txt_gate.last_g.mean().item())

    if gate_means:
        print(f"[train_one_epoch_i2t_only] gate mean over epoch: {sum(gate_means)/len(gate_means):.4f}")
    return total_loss / max(num_steps, 1)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 1e-4 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train_pretrained_file = './pretrained_embeddings/MSCOCO/clip_embeddings_with_category.pt'
    val_pretrained_file = './pretrained_embeddings/MSCOCO/clip_embeddings_with_category_val.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device))
    num_epochs = 60
    batch_size = 512
    topk = 5
    # After TPOneHot retrieval, apply hash-bucketing coreset + vote-topk (ARROW-style)
    distance = 'HashCoreset'  # 'TPOneHot' (baseline) or 'HashCoreset' (TPOneHot + coreset)
    coreset_ratio = 0.8
    coreset_max_prefix_len = 20
    coreset_seed = 0
    hashcoreset_diagnostics = False
    # Set to False to skip reciprocal rerank and pass raw TPOneHot top-k directly
    # into HashCoreset.  The rerank code is preserved and can be re-enabled here.
    use_reciprocal_rerank = True
    # Loss function: 'rcm' or 'contrastive'
    #   'rcm'        : RCMLoss  — bounded [0,1], robust to noisy pseudo pairs
    #   'contrastive': ContrastiveLoss — hard-margin triplet, max_violation=True
    loss_type = 'rcm'

    data = torch.load(train_pretrained_file)
    val_data = torch.load(val_pretrained_file)
    print("train data length:", len(data))
    seed_torch()
    random.shuffle(data)
    split_idx = int(0.8 * len(data))

    train_set, test_set = data[:split_idx], data[split_idx:]
    aligned_train_set = EmbeddingDataset(train_set, mode='aligned')
    aligned_test_set = EmbeddingDataset(test_set, mode='aligned')
    aligned_val_set = EmbeddingDataset(val_data, mode='aligned')
    unaligned_train_set = EmbeddingDataset(train_set, mode='unaligned')

    # ========== 构建伪对齐数据集 ==========
    construct_time = time.time()
    prune_train_to_coreset = (distance == 'HashCoreset')
    if prune_train_to_coreset:
        print(f"Enable representative-only training set pruning for `{distance}`.")

    def _build_bidir(dataset, tag):
        if distance == 'HashCoreset':
            i2t_data, t2i_data = build_bidirectional_pseudo_aligned_datasets_hashcoreset_reciprocal(
                dataset,
                topk=topk,
                device=str(device),
                coreset_ratio=coreset_ratio,
                coreset_max_prefix_len=coreset_max_prefix_len,
                coreset_seed=coreset_seed,
                prune_to_coreset_reps=prune_train_to_coreset,
                hashcoreset_diagnostics=hashcoreset_diagnostics,
                use_reciprocal_rerank=use_reciprocal_rerank,
            )
        else:
            i2t_data = build_pseudo_aligned_dataset(
                dataset, distance=distance, topk=topk,
                device=str(device),
                coreset_ratio=coreset_ratio,
                coreset_max_prefix_len=coreset_max_prefix_len,
                coreset_seed=coreset_seed,
                prune_to_coreset_reps=prune_train_to_coreset,
                hashcoreset_diagnostics=hashcoreset_diagnostics,
            )
            t2i_data = build_pseudo_aligned_dataset_t2i(
                dataset, distance=distance, topk=topk,
                device=str(device),
                coreset_ratio=coreset_ratio,
                coreset_max_prefix_len=coreset_max_prefix_len,
                coreset_seed=coreset_seed,
                prune_to_coreset_reps=prune_train_to_coreset,
                hashcoreset_diagnostics=hashcoreset_diagnostics,
            )
        i2t_set = EmbeddingDataset(i2t_data, mode='aligned')
        t2i_set = EmbeddingDataset(t2i_data, mode='aligned')
        elapsed = time.time() - construct_time
        print(f'construct time ({tag} bidir): {elapsed:.1f}s')
        print(f"  queries kept (i2t): {len(i2t_set)}/{len(dataset)}")
        print(f"  queries kept (t2i): {len(t2i_set)}/{len(dataset)}")
        return i2t_set, t2i_set

    pseudo_aligned_train_set_i2t, pseudo_aligned_train_set_t2i = _build_bidir(unaligned_train_set, 'train')

    construct_time = time.time()
    pseudo_aligned_test_set_i2t, pseudo_aligned_test_set_t2i = _build_bidir(aligned_test_set, 'test')

    construct_time = time.time()
    pseudo_aligned_val_set_i2t, pseudo_aligned_val_set_t2i = _build_bidir(aligned_val_set, 'val')

    # Dataloaders
    train_loader_i2t = DataLoader(pseudo_aligned_train_set_i2t, batch_size=batch_size, shuffle=True)
    test_loader_i2t = DataLoader(pseudo_aligned_test_set_i2t, batch_size=batch_size, shuffle=False)
    val_loader_i2t = DataLoader(pseudo_aligned_val_set_i2t, batch_size=batch_size, shuffle=False)

    train_loader_t2i = DataLoader(pseudo_aligned_train_set_t2i, batch_size=batch_size, shuffle=True)
    test_loader_t2i = DataLoader(pseudo_aligned_test_set_t2i, batch_size=batch_size, shuffle=False)
    val_loader_t2i = DataLoader(pseudo_aligned_val_set_t2i, batch_size=batch_size, shuffle=False)

    pseudo_aligned_model = PseudoAlignModel(embed_dim=512).to(device)
    pseudo_aligned_optimizer = torch.optim.Adam(pseudo_aligned_model.parameters(), lr=1e-4)

    def _make_criterion():
        if loss_type == 'contrastive':
            return ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True).to(device)
        return RCMLoss(tau=0.07).to(device)

    pseudo_aligned_criterion = _make_criterion()

    # ========== 选择训练模式 ==========
    # 可选值: 'i2t_only', 't2i_only', 'bidir'
    TRAINING_MODE = 'bidir'
    # 训练日程：
    #   - 'bidir'    : 前半 i2t、后半 t2i
    #   - 'i2t_full' : 全部 epoch 都用 i2t
    #   - 't2i_full' : 全部 epoch 都用 t2i
    TRAINING_SCHEDULE = 'bidir'

    if TRAINING_MODE == 'i2t_only':
        print("=" * 60)
        print("Training mode: I2T-ONLY (image as query, top-k texts as key/value)")
        print("=" * 60)
        model = I2TOnlyModel(embed_dim=512).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = _make_criterion()
    elif TRAINING_MODE == 't2i_only':
        print("=" * 60)
        print("Training mode: T2I-ONLY (text as query, top-k images as key/value)")
        print("=" * 60)
        model = T2IOnlyModel(embed_dim=512).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = _make_criterion()
    else:  # bidir
        print("=" * 60)
        print("Training mode: BIDIRECTIONAL (both i2t and t2i)")
        print("=" * 60)
        model = pseudo_aligned_model
        optimizer = pseudo_aligned_optimizer
        criterion = pseudo_aligned_criterion
        t2i_weight = 2.0

    train_time_total = 0.0
    train_eval_total_begin = time.time()

    for epoch in range(num_epochs):
        epoch_train_begin = time.time()
        if TRAINING_SCHEDULE == 'i2t_full':
            train_loss = train_one_epoch(
                model, train_loader_i2t, optimizer, criterion, device, mode='i2t')
            phase = 'i2t-only'
            phase_epoch = epoch
        elif TRAINING_SCHEDULE == 't2i_full':
            train_loss = train_one_epoch(
                model, train_loader_t2i, optimizer, criterion, device, mode='t2i')
            phase = 't2i-only'
            phase_epoch = epoch
        else:
            # bidir：前半 i2t，后半 t2i，并在切换时重置优化器
            if epoch < num_epochs // 2:
                train_loss = train_one_epoch(
                    model, train_loader_i2t, optimizer, criterion, device, mode='i2t')
                phase = 'i2t-only'
                phase_epoch = epoch
            else:
                if epoch == num_epochs // 2:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                train_loss = train_one_epoch(
                    model, train_loader_t2i, optimizer, criterion, device, mode='t2i')
                phase = 't2i-only'
                phase_epoch = epoch - num_epochs // 2
        train_time_total += time.time() - epoch_train_begin

        # 在 test 和 val 集上分别计算融合指标
        metrics_fused_test = evaluate_retrieval_fused(
            model, test_loader_i2t, test_loader_t2i, device,
            img_fuse_weight=0.5, txt_fuse_weight=0.5, l2_normalize_fused=True)
        metrics_fused_val = evaluate_retrieval_fused(
            model, val_loader_i2t, val_loader_t2i, device,
            img_fuse_weight=0.5, txt_fuse_weight=0.5, l2_normalize_fused=True)

        i2t_branch_metrics_test = {k: v for k, v in metrics_fused_test.items() if "_i2tBranch_" in k}
        t2i_branch_metrics_test = {k: v for k, v in metrics_fused_test.items() if "_t2iBranch_" in k}

        # 计算两个分支在 i2t / t2i 方向的平均值
        avg_metrics_test = {}
        avg_metrics_val = {}
        metric_names = ['R@1', 'R@5', 'R@10', 'MRR', 'mAP']
        for mn in metric_names:
            for mf, avg_d in [(metrics_fused_test, avg_metrics_test), (metrics_fused_val, avg_metrics_val)]:
                k1_i2t = f'{mn}_i2tBranch_i2t'; k2_i2t = f'{mn}_t2iBranch_i2t'
                k1_t2i = f'{mn}_i2tBranch_t2i'; k2_t2i = f'{mn}_t2iBranch_t2i'
                if k1_i2t in mf and k2_i2t in mf:
                    avg_d[f'{mn}_avg_i2t'] = (mf[k1_i2t] + mf[k2_i2t]) / 2
                if k1_t2i in mf and k2_t2i in mf:
                    avg_d[f'{mn}_avg_t2i'] = (mf[k1_t2i] + mf[k2_t2i]) / 2

        i2t_branch_metrics_val = {k: v for k, v in metrics_fused_val.items() if "_i2tBranch_" in k}
        t2i_branch_metrics_val = {k: v for k, v in metrics_fused_val.items() if "_t2iBranch_" in k}

        print(f"Epoch {epoch + 1} [{phase}] | Train Loss: {train_loss:.4f}")
        print(f"  [test] i2tBranch: {i2t_branch_metrics_test}")
        print(f"  [test] t2iBranch: {t2i_branch_metrics_test}")
        print(f"  [test] Average:   {avg_metrics_test}")
        print(f"  [val]  i2tBranch: {i2t_branch_metrics_val}")
        print(f"  [val]  t2iBranch: {t2i_branch_metrics_val}")
        print(f"  [val]  Average:   {avg_metrics_val}")

        # 阶段式 LR 衰减：每个阶段各自 0/10/20... 计数
        lr = 1e-4 * (0.1 ** (phase_epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print('training time (train only):', train_time_total)
    print('training time (train + eval total):', time.time() - train_eval_total_begin)
