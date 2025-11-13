"""
Orthogonal LSH + Private Hamming + Cross Attention-based fusion
"""
import torch
import random
import time



from data_loader.MSCOCODataset import EmbeddingDataset, embedding_collate_fn
from torch.utils.data import  DataLoader
from data_loader.utils import build_pseudo_aligned_dataset, build_missing_pseudo_aligned_dataset
from utils.utils import seed_torch
from model.models import  ContrastiveLoss
from model.FedTModel import FedT
import time


def train_one_epoch(model, dataloader,  optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)

        img_feat, txt_feat = model(img, txt)
        loss = criterion(img_feat, txt_feat)
        #loss = criterion(img_feat, txt_feat, is_matched=is_matched, category=category)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# def evaluate_retrieval(model, dataloader, device='cuda'):
#     model.eval()
#     all_img, all_txt = [], []
#
#     for batch in dataloader:
#         img = batch['image_embedding'].to(device)
#         txt = batch['text_embedding'].to(device)
#         img_feat, txt_feat = model(img, txt)
#         all_img.append(img_feat)
#         all_txt.append(txt_feat)
#
#     img_mat = torch.cat(all_img, dim=0)
#     txt_mat = torch.cat(all_txt, dim=0)
#     sim = img_mat @ txt_mat.T  # cosine similarity
#
#     def recall_at_k(sim, k):
#         N = sim.size(0)
#         targets = torch.arange(N, device=sim.device)
#         _, topk_i2t = sim.topk(k, dim=1)
#         _, topk_t2i = sim.topk(k, dim=0)
#         recall_i2t = (topk_i2t == targets[:, None]).any(dim=1).float().mean().item()
#         recall_t2i = (topk_t2i == targets[None, :]).any(dim=0).float().mean().item()
#         return recall_i2t, recall_t2i
#
#     metrics = {}
#     for k in [1, 5, 10]:
#         i2t, t2i = recall_at_k(sim, k)
#         metrics[f'R@{k}_i2t'] = i2t
#         metrics[f'R@{k}_t2i'] = t2i
#     return metrics



def evaluate_retrieval(model, dataloader, device='cuda'):
    model.eval()
    all_img, all_txt = [], []

    for batch in dataloader:
        img = batch['image_embedding'].to(device)
        txt = batch['text_embedding'].to(device)
        img_feat, txt_feat = model(img, txt)
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



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 1e-4 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def add_gaussian_noise_to_embeddings(data_list, noise_level=0.01):
    """
    为列表中每个元素的text_embedding添加高斯噪音

    参数:
    data_list: 包含嵌入向量的字典列表
    noise_level: 高斯噪音的标准差，控制噪音强度

    返回:
    添加噪音后的数据列表
    """
    import torch

    noisy_data_list = []

    for item in data_list:
        # 获取原始文本嵌入
        original_text_emb = item['text_embedding']

        # 生成与原始嵌入形状相同的高斯噪音
        noise = torch.randn_like(original_text_emb) * noise_level

        # 添加噪音到文本嵌入
        noisy_text_emb = original_text_emb + noise

        # 创建新的字典，所有其他字段保持不变，只有text_embedding更新
        noisy_item = {
            'image_embedding': item['image_embedding'],
            'text_embedding': noisy_text_emb,
            'category': item['category'] if 'category' in item else torch.tensor(-1)
        }

        # 如果原始项中有其他键，也复制到新项中
        for key in item:
            if key not in noisy_item:
                noisy_item[key] = item[key]

        noisy_data_list.append(noisy_item)

    return noisy_data_list

if __name__ == '__main__':
    pretrained_file = '/data2/kudret/codes/FedML/pretrained_embeddings/Flicker30k/clip_embeddings.pt'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_epochs = 60
    batch_size = 512
    topk = 5
    data = torch.load(pretrained_file)
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
    original_pseudo_aligned_train_set = build_pseudo_aligned_dataset(unaligned_train_set, distance='Euclidean', topk=topk)
    # Add Gaussian noise to text embeddings
    noise_pseudo_aligned_train_set = add_gaussian_noise_to_embeddings(original_pseudo_aligned_train_set, noise_level=0.1)

    pseudo_aligned_train_set = EmbeddingDataset(noise_pseudo_aligned_train_set, mode='aligned')
    pseudo_aligned_train_loader = DataLoader(pseudo_aligned_train_set, batch_size=batch_size, shuffle=True)

    original_pseudo_aligned_test_set = build_pseudo_aligned_dataset(unaligned_test_set, distance='Euclidean', topk=topk)
    noise_pseudo_aligned_test_set = add_gaussian_noise_to_embeddings(original_pseudo_aligned_test_set, noise_level=0.1)
    pseudo_aligned_test_set = EmbeddingDataset(noise_pseudo_aligned_test_set, mode='aligned')

    pseudo_aligned_test_loader = DataLoader(pseudo_aligned_test_set, batch_size=batch_size,
                                                    shuffle=False)
    print('construct time:', time.time() - construct_time)

    pseudo_aligned_model = FedT(embed_dim=512).to(device)
    pseudo_aligned_optimizer = torch.optim.Adam(pseudo_aligned_model.parameters(), lr=1e-4)
    pseudo_aligned_criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
    pseudo_aligned_criterion.to(device)
    train_begin = time.time()
    for epoch in range(num_epochs):
        adjust_learning_rate(pseudo_aligned_optimizer, epoch)
        # Train pseudo aligned model
        pseudo_aligned_train_loss = train_one_epoch(pseudo_aligned_model, pseudo_aligned_train_loader,
                                                    pseudo_aligned_optimizer, pseudo_aligned_criterion, device)
        pseudo_aligned_test_metrics = evaluate_retrieval(pseudo_aligned_model, pseudo_aligned_test_loader,
                                                         device)
        print(
            f"Epoch {epoch + 1} | Pseudo Aligned Loss: {pseudo_aligned_train_loss:.4f} | {pseudo_aligned_test_metrics}")

    print('training time:', time.time() - train_begin)



