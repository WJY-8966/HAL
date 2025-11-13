"""
Orthogonal LSH + Private Hamming + Cross Attention-based fusion
"""
import torch
import random
from data_loader.MSCOCODataset import EmbeddingDataset, embedding_collate_fn
from torch.utils.data import  DataLoader
from utils.utils import seed_torch
from model.models import ClipLinear, ContrastiveLoss
from model.FedSimModel import FedSimModel
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

if __name__ == '__main__':
    pretrained_file = '/data2/kudret/codes/FedML/pretrained_embeddings/Flicker30k/clip_embeddings.pt'
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    num_epochs = 60
    batch_size = 512
    topk = 5
    data = torch.load(pretrained_file)
    seed_torch()
    seed_torch()
    for item in data:
        item['category'] = torch.tensor([-1], dtype=torch.long)  # Add dummy category
    random.shuffle(data)
    split_idx = int(0.8 * len(data))

    train_set, test_set = data[:split_idx], data[split_idx:]
    aligned_train_set = EmbeddingDataset(train_set, mode='aligned')
    aligned_test_set = EmbeddingDataset(test_set, mode='aligned')



    aligned_train_loader = DataLoader(aligned_train_set, batch_size=batch_size, shuffle=True)
    aligned_test_loader = DataLoader(aligned_test_set, batch_size=batch_size,
                                                    shuffle=False)

    aligned_model =ClipLinear(embed_dim=512).to(device)
    aligned_optimizer = torch.optim.Adam(aligned_model.parameters(), lr=1e-4)
    aligned_criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
    aligned_criterion.to(device)

    train_begin = time.time()
    for epoch in range(num_epochs):
        adjust_learning_rate(aligned_optimizer, epoch)
        # Train aligned model
        aligned_train_loss = train_one_epoch(aligned_model, aligned_train_loader,
                                                    aligned_optimizer, aligned_criterion, device)
        aligned_test_metrics = evaluate_retrieval(aligned_model, aligned_test_loader,
                                                         device)
        print(
            f"Epoch {epoch + 1} | Aligned Loss: {aligned_train_loss:.4f} | {aligned_test_metrics}")


    print(f"Training Time: {time.time() - train_begin}")

