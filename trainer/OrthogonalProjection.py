import torch
import torch.nn.functional as F


def estimate_shift(image_embeddings, text_embeddings):
    """
    estimate image and text embedding mean difference（shift）
    """
    assert image_embeddings.shape == text_embeddings.shape
    shift = (image_embeddings - text_embeddings).mean(dim=0)
    shift = shift / shift.norm()  # 单位化
    return shift  # shape: [D]

def generate_random_lsh_vectors(dim, num_vecs):
    """
    generate num_vecs random unit vectors in D-dimensional space

    Args:
        dim (int):
        num_vecs (int):

    Returns:
        torch.Tensor: shape [num_vecs, dim]
    """
    rand_vecs = torch.randn(num_vecs, dim).to('cuda:0')  # generate random vectors
    rand_vecs = F.normalize(rand_vecs, dim=1)  # L2 normalize
    return rand_vecs

def generate_orthogonal_lsh_projections(shift, dim, num_vecs):
    """
    generate num_vecs orthogonal vectors to the shift vector in D-dimensional space
    """
    shift = shift / shift.norm()
    basis = []

    for _ in range(num_vecs):
        r = torch.randn(dim).to(shift.device)
        # project r onto the orthogonal complement of shift
        r_proj = r - (r @ shift) * shift
        r_proj = r_proj / r_proj.norm()
        basis.append(r_proj)

    return torch.stack(basis)  # shape: [num_vecs, dim]

def lsh_hash_bits(x, r):
    """
    x: [N, D], r: [K, D]
    :return [N, K] binary {0, 1}
    """
    proj = x @ r.T
    bits = ((proj > 0).int())  # sign(x @ r): >0 → 1, ≤0 → 0
    return bits

def normalize_embeddings(embeddings):
    """
    L2 normalize embeddings along the feature dimension
    """
    return F.normalize(embeddings, dim=1)


def generate_lsh_embeddings(image_embeddings, text_embeddings, num_vecs=512, orthogonal=True):
    """
    generate LSH embeddings for image and text embeddings
    """
    img_embeddings = normalize_embeddings(image_embeddings)
    txt_embeddings = normalize_embeddings(text_embeddings)
    shift = estimate_shift(img_embeddings, txt_embeddings)
    if orthogonal:
        lsh_projections = generate_orthogonal_lsh_projections(shift, dim=img_embeddings.size(1), num_vecs=num_vecs)
    else:
        lsh_projections = generate_random_lsh_vectors(dim=img_embeddings.size(1), num_vecs=num_vecs)
    img_bins = lsh_hash_bits(img_embeddings, lsh_projections)
    txt_bins = lsh_hash_bits(txt_embeddings, lsh_projections)
    return img_bins, txt_bins