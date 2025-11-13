import numpy as np
from tqdm import tqdm
import torch
from scipy.sparse import lil_matrix, csr_matrix
from trainer.OrthogonalProjection import lsh_hash_bits,generate_random_lsh_vectors
import torch.nn.functional as F




def generate_disjoint_hashes(n, r, m, seed=42):
    """
    Generate disjoint Bloom-style hash sets H0(i), H1(i) for each i in [n].
    """
    np.random.seed(seed)
    all_indices = np.arange(m)
    used_indices = set()
    H0, H1 = {}, {}
    for i in tqdm(range(n)):
        available = list(set(all_indices) - used_indices)
        H0[i] = np.random.choice(available, r, replace=False)
        used_indices.update(H0[i])
        available = list(set(all_indices) - used_indices)
        H1[i] = np.random.choice(available, r, replace=False)
        used_indices.update(H1[i])
    return H0, H1

def encode_matrix(X, H0, H1, m):
    """
    Encode a matrix X (N x n) using Bloom-style disjoint hashes.
    """
    encoded = np.zeros((X.shape[0], m), dtype=int)
    for idx, x in enumerate(X):
        h = np.zeros(m, dtype=int)
        for i, bit in enumerate(x):
            indices = H0[i] if bit == 0 else H1[i]
            h[indices] = 1
        encoded[idx] = h
    return encoded

def encode_matrix_torch(X, H0, H1, m):
    """
    Encode binary matrix X (N x n) into Bloom-style disjoint hashes (N x m).
    Returns torch.Tensor on specified device.
    """
    N, n = X.shape
    encoded = torch.zeros((N, m), dtype=torch.uint8, device=X.device)

    for i in range(n):
        idx0 = (X[:, i] == 0).nonzero(as_tuple=True)[0]
        idx1 = (X[:, i] == 1).nonzero(as_tuple=True)[0]
        if len(idx0) > 0:
            encoded[idx0[:, None], torch.tensor(H0[i], device=X.device)] = 1
        if len(idx1) > 0:
            encoded[idx1[:, None], torch.tensor(H1[i], device=X.device)] = 1

    return encoded

def encode_matrix_chunked(X_bin, H0, H1, m, batch_size=1024, device='cuda'):
    """
    Encode a binary matrix (X_bin) using Bloom-style disjoint hashes on GPU.
    X_bin: (N, n) binary tensor
    H0, H1: dict of disjoint hash indices
    m: total hash length
    Returns: (N, m) binary encoded tensor
    """
    N, n = X_bin.shape
    X_bin = X_bin.to(device)
    encoded = torch.zeros((N, m), dtype=torch.uint8, device=device)

    # convert H0/H1 to tensors of shape (n, r)
    r = len(next(iter(H0.values())))
    H0_tensor = torch.tensor([H0[i] for i in range(n)], dtype=torch.long, device=device)
    H1_tensor = torch.tensor([H1[i] for i in range(n)], dtype=torch.long, device=device)

    for start in tqdm(range(0, N, batch_size), desc='Encoding'):
        end = min(start + batch_size, N)
        batch = X_bin[start:end]  # (B, n)
        B = batch.shape[0]

        # Prepare indexing: for each bit, choose from H0 or H1
        batch = batch.unsqueeze(-1)  # (B, n, 1)
        H_indices = torch.where(batch == 0, H0_tensor, H1_tensor)  # (B, n, r)
        H_indices = H_indices.view(B, -1)  # (B, n*r)

        # Scatter into encoded
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n * r)
        encoded[start:end].scatter_(1, H_indices, 1)

    return encoded

def bit_flip_matrix(X, epsilon, seed=42):
    """
    Apply bit-flipping to each vector in matrix X independently.
    """
    np.random.seed(seed)
    flip_mask = np.random.rand(*X.shape) < epsilon
    return np.bitwise_xor(X, flip_mask.astype(int))

def bit_flip_matrix_torch(X, epsilon, seed=42):
    torch.manual_seed(seed)
    noise = torch.rand_like(X.float(), device=X.device) < epsilon
    return (X ^ noise.bool()).to(torch.uint8)

def bit_flip_ldp(X, epsilon, seed=42):
    np.random.seed(seed)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # flip prob
    flip_mask = np.random.rand(*X.shape)
    noise = (flip_mask > p).astype(int)  # flip when > p
    return np.bitwise_xor(X, noise)

def corrected_hamming_matrix(A_enc, B_enc, alpha, phi, m):
    """
    Compute corrected Hamming distance between every pair of vectors in A and B.
    """
    N, M = A_enc.shape[0], B_enc.shape[0]
    D = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            raw_diff = np.sum(A_enc[i] ^ B_enc[j])
            D[i, j] = (raw_diff - alpha * m) / (phi * (1 - alpha))
    return D

def corrected_hamming_matrix_torch(A_enc, B_enc, alpha, phi, m):
    """
    Compute corrected Hamming distance between each image and all texts.
    """
    # Raw noisy Hamming distance
    # XOR operation (bitwise difference), then sum across bit-dimension
    raw_diff = (A_enc.unsqueeze(1) ^ B_enc.unsqueeze(0)).sum(dim=2)  # shape (B, T)
    print("Raw Hamming distance shape:", raw_diff.shape)
    corrected = (raw_diff - alpha * m) / (phi * (1 - alpha))
    return corrected

import torch

def corrected_hamming_distance_chunked(A_enc, B_enc, alpha, phi, m, batch_size_img=512, batch_size_txt=1024):
    """
    Compute corrected Hamming distance in chunks between A_enc (images) and B_enc (texts).
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

            # Apply correction
            corrected = (raw_diff - alpha * m) / (phi * (1 - alpha))  # shape (B, T)

            # Assign to output
            D[i_start:i_end, j_start:j_end] = corrected

    return D  # shape (num_images, num_texts)

def private_data_transform(X, r=5, epsilon=0.1, seed=42):
    """
    Transform data X using Bloom-style encoding and bit-flipping.
    X: binary matrix (N x n)
    Returns: private transformed matrix (N x m)
    """
    X = F.normalize(X, dim=1)
    lsh_projections = generate_random_lsh_vectors(dim=X.size(1), num_vecs=512).to(X.device)
    X_bins = lsh_hash_bits(X, lsh_projections)
    n = X.shape[1]

    # Step 1: Build hashes
    m = int(r * 1024)
    H0, H1 = generate_disjoint_hashes(n, r, m, seed=seed)

    # Step 2: Encode and perturb
    X_enc = encode_matrix_torch(X_bins, H0, H1, m)
    X_noisy = bit_flip_matrix_torch(X_enc, epsilon, seed=seed)

    return X_noisy


def private_knn(A, B, k, r=10, m=128, epsilon=0.1, seed=42):
    """
    Compute private KNN using Bloom-style encoding and bit-flipping.
    A: query set (N x n), B: database set (M x n)
    Returns indices of top-k neighbors in B for each row in A.
    """
    n = A.shape[1]
    alpha = 2 * epsilon * (1 - epsilon)
    phi = 2 * r

    # Step 1: Build hashes
    H0, H1 = generate_disjoint_hashes(n, r, m, seed=seed)

    # Step 2: Encode and perturb
    A_enc = encode_matrix(A, H0, H1, m)
    B_enc = encode_matrix(B, H0, H1, m)
    A_noisy = bit_flip_matrix(A_enc, epsilon, seed=seed)
    B_noisy = bit_flip_matrix(B_enc, epsilon, seed=seed)

    # Step 3: Corrected Hamming distances
    dist = corrected_hamming_matrix(A_noisy, B_noisy, alpha, phi, m)

    # Step 4: Get top-k indices
    knn_indices = np.argsort(dist, axis=1)[:, :k]
    return knn_indices, dist

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.randint(0, 2, (3, 5))  # 3 query samples, 5 bits
    B = np.random.randint(0, 2, (6, 5))  # 6 database samples, 5 bits
    A = torch.tensor(A)
    B = torch.tensor(B)
    n = A.shape[1]
    r = 1
    m = 10
    H0, H1 = generate_disjoint_hashes(n, r, m, seed=42)
    print(A)
    print(H0, H1)
    print(encode_matrix_torch(A, H0, H1, m))

    A_chunk = torch.tensor([[1, 1, 1, 1],
                            [0, 1, 0, 0]], dtype=torch.uint8)  # 2个样本
    B_chunk = torch.tensor([[1, 1, 1, 1],
                            [0, 0, 1, 0],
                            [1, 0, 1, 0]], dtype=torch.uint8)  # 3个样本
    raw_diff = (A_chunk.unsqueeze(1) ^ B_chunk.unsqueeze(0)).sum(dim=2)
    print(raw_diff)

    # k = 2
    # indices, dists = private_knn(A, B, k)
    # print("A:")
    # print(A)
    # print("B:")
    # print(B)
    # print("Top-k neighbor indices in B for each row in A:")
    # print(indices)
    # print("Estimated distances:")
    # print(np.round(dists, 2))
