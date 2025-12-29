import torch

def generate_tpoh_hashes(n, seed=42):
    # 前半 [0..n-1] 给 0 类，后半 [n..2n-1] 给 1 类；各自打乱保证公开随机
    g = torch.Generator().manual_seed(seed)
    H0 = torch.randperm(n, generator=g).cpu().numpy()
    H1 = (n + torch.randperm(n, generator=g)).cpu().numpy()
    return {i: int(H0[i]) for i in range(n)}, {i: int(H1[i]) for i in range(n)}, 2*n

@torch.no_grad()
def encode_tpoh_torch(X, H0, H1, m):
    # X:(N,n)->B:(N,2n) one-hot 按位；无碰撞
    N, n = X.shape
    B = torch.zeros((N, m), dtype=torch.uint8, device=X.device)
    for i in range(n):
        idx0 = (X[:, i]==0).nonzero(as_tuple=True)[0]
        idx1 = (X[:, i]==1).nonzero(as_tuple=True)[0]
        if idx0.numel()>0: B[idx0, H0[i]] = 1
        if idx1.numel()>0: B[idx1, H1[i]] = 1
    return B

def compute_hamming_distance_chunked(A_enc, B_enc, batch_size_img=512, batch_size_txt=1024):
    """
    Compute Hamming distance in chunks between A_enc (images) and B_enc (texts).
    Returns a (num_images, num_texts) matrix of corrected distances.
    """
    num_images = A_enc.shape[0]
    num_texts = B_enc.shape[0]
    D = torch.zeros((num_images, num_texts), device=A_enc.device)

    for i_start in range(0, num_images, batch_size_img):
        i_end = min(i_start + batch_size_img, num_images)
        A_chunk = A_enc[i_start:i_end]  # shape (B, m)

        for j_start in range(0, num_texts, batch_size_txt):
            j_end = min(j_start + batch_size_txt, num_texts)
            B_chunk = B_enc[j_start:j_end]  # shape (T, m)

            # Compute raw Hamming distance in chunk
            raw_diff = (A_chunk.unsqueeze(1) ^ B_chunk.unsqueeze(0)).sum(dim=2)  # shape (B, T)

            # Assign to output
            D[i_start:i_end, j_start:j_end] = raw_diff

    return D  # shape (num_images, num_texts)