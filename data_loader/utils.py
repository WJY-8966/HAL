import os
import sys
sys.path.append('/data2/kudret/codes/FedML')
import torch
from torch.utils.data import  DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import json
from imagebind.models.imagebind_model import ModalityType
from trainer.PrivateHamming import (encode_matrix, bit_flip_matrix, corrected_hamming_matrix, generate_disjoint_hashes,
                                    encode_matrix_torch,encode_matrix_chunked, bit_flip_matrix_torch,
                                    corrected_hamming_matrix_torch, corrected_hamming_distance_chunked)
from trainer.OrthogonalProjection import (generate_orthogonal_lsh_projections,estimate_shift,
                                          lsh_hash_bits, normalize_embeddings, generate_random_lsh_vectors,
                                          generate_lsh_embeddings)
from trainer.TPOneHot import (generate_tpoh_hashes,encode_tpoh_torch,compute_hamming_distance_chunked)

@torch.no_grad()
def save_MSCOCO_imagebind_embeddings(dataset, encoder, save_path, device='cuda:6', batch_size=512, collate_fn=None):
    """
     ImageBind embedding save as .pt

    Args:
        dataset: CocoCaptionDataset
        imagebind_model: ImageBind  model
        save_path: embedding save path（.pt）
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_data = []

    for batch in tqdm(dataloader, desc="Encoding with ImageBind"):
        images = batch['img_path']
        texts = batch['caption']
        categories = batch['category']

        embeddings = encoder(images=images, texts=texts)

        image_embeds = embeddings[ModalityType.VISION]   # [B, D]
        text_embeds = embeddings[ModalityType.TEXT]      # [B, D]

        for i in range(len(images)):
            all_data.append({
                'img_path': batch['img_path'][i],
                'caption': batch['caption'][i],
                'image_embedding': image_embeds[i].cpu(),
                'text_embedding': text_embeds[i].cpu(),
                'category': categories[i]
            })

    torch.save(all_data, save_path)
    print(f"[✓] Saved {len(all_data)} embeddings to {save_path}")

def save_MSCOCO_clip_embeddings(dataset, encoder, save_path, batch_size=512, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_data = []
    for batch in tqdm(dataloader, desc="Encoding with CLIP"):
        images = batch['image']
        captions = batch['caption']
        img_paths = batch['img_path']
        categories = batch['category']


        image_embeds, text_embeds = encoder(images, captions)

        for i in range(len(images)):
            all_data.append({
                'image_embedding': image_embeds[i].cpu(),
                'text_embedding': text_embeds[i].cpu(),
                'caption': captions[i],
                'img_path': img_paths[i],
                'category': categories[i]
            })

    torch.save(all_data, save_path)
    print(f"Saved {len(all_data)} samples to {save_path}")

@torch.no_grad()
def save_Flicker30k_imagebind_embeddings(dataset, encoder, save_path, device='cuda', batch_size=512, collate_fn=None):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_data = []

    for batch in tqdm(dataloader, desc="Encoding with ImageBind"):
        images = batch['img_path']
        texts = batch['caption']

        embeddings = encoder(images=images, texts=texts)

        image_embeds = embeddings[ModalityType.VISION]   # [B, D]
        text_embeds = embeddings[ModalityType.TEXT]      # [B, D]

        for i in range(len(images)):
            all_data.append({
                'img_path': batch['img_path'][i],
                'caption': batch['caption'][i],
                'image_embedding': image_embeds[i].cpu(),
                'text_embedding': text_embeds[i].cpu()
            })

    torch.save(all_data, save_path)
    print(f"[✓] Saved {len(all_data)} embeddings to {save_path}")



def save_Flicker30k_clip_embeddings(dataset, encoder, save_path, batch_size=512, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_data = []
    for batch in tqdm(dataloader, desc="Encoding with CLIP"):
        images = batch['image']
        captions = batch['caption']
        img_paths = batch['img_path']

        image_embeds, text_embeds = encoder(images, captions)

        for i in range(len(images)):
            all_data.append({
                'image_embedding': image_embeds[i].cpu(),
                'text_embedding': text_embeds[i].cpu(),
                'caption': captions[i],
                'img_path': img_paths[i],
            })

    torch.save(all_data, save_path)
    print(f"Saved {len(all_data)} samples to {save_path}")


def save_IEMOCAP_imagebind_embeddings(dataset, encoder, save_path, device='cuda', batch_size=512, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_data = []

    for batch in tqdm(dataloader, desc="Encoding with ImageBind"):
        videos = batch['video_path']
        texts = batch['caption']
        audios = batch['audio_path']

        embeddings = encoder(images=None, texts=texts, audios=audios, videos=videos)

        video_embeds = embeddings[ModalityType.VISION]  # [B, D]
        text_embeds = embeddings[ModalityType.TEXT]  # [B, D]
        audio_embeds = embeddings[ModalityType.AUDIO]  # [B, D]
        print(video_embeds.shape)
        for i in range(len(videos)):
            print("Processing sample:", i)
            all_data.append({
                'video_path': batch['video_path'][i],
                'audio_path': batch['audio_path'][i],
                'caption': batch['caption'][i],
                'video_embedding': video_embeds[i].cpu(),
                'text_embedding': text_embeds[i].cpu(),
                'audio_embedding': audio_embeds[i].cpu()
            })

    torch.save(all_data, save_path)
    print(f"[✓] Saved {len(all_data)} embeddings to {save_path}")

def save_embeddings(dataset, image_encoder, text_encoder, save_path, collate_fn, batch_size=256, device='cuda:6'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    image_encoder.eval()
    text_encoder.eval()

    all_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding..."):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            image_embeds = image_encoder(images)          # [B, D]
            text_embeds = text_encoder(input_ids, attention_mask)  # [B, D]

            for i in range(images.size(0)):
                all_data.append({
                    'img_path': batch['img_paths'][i],
                    'caption': batch['captions'][i],
                    'image_embedding': image_embeds[i].cpu(),
                    'text_embedding': text_embeds[i].cpu(),
                })

    torch.save(all_data, save_path)
    print(f"Saved {len(all_data)} embeddings to {save_path}")


def extract_image_id(img_path):
    filename = os.path.basename(img_path)
    return int(os.path.splitext(filename)[0].split('_')[-1])


def build_pseudo_aligned_avg_topk_dataset(pseudo_aligned_data):
    new_dataset = []
    for item in pseudo_aligned_data:
        avg_text_emb = item['text_embedding'].mean(dim=0)  # [D]
        new_dataset.append({
            'image_embedding': item['image_embedding'],  # [D]
            'text_embedding': avg_text_emb,  # [D]
            'category': item.get('category', torch.tensor(-1))  # 可选
        })
    return new_dataset


def build_pseudo_aligned_dataset_weight(unaligned_data, top_k=5, weighted=True, distance='cosine'):
    image_embeddings = torch.stack([item['image_embedding'] for item in unaligned_data])
    text_embeddings = torch.stack([item['text_embedding'] for item in unaligned_data])
    categories = [item['category'] for item in unaligned_data]

    # normalize
    image_embeddings = F.normalize(image_embeddings.float(), dim=1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=1)

    # select distance metric
    if distance.lower() == 'cosine':
        # cosine similarity
        sim_matrix = image_embeddings @ text_embeddings.T  # [N, N]
        select_fn = torch.topk
    elif distance.lower() == 'hamming':
        # img_bin = (image_embeddings > 0).float()
        # txt_bin = (text_embeddings > 0).float()
        img_bin, txt_bin = generate_lsh_embeddings(image_embeddings, text_embeddings)
        hamming_dist = (img_bin.unsqueeze(1) != txt_bin.unsqueeze(0)).sum(dim=2).float()
        sim_matrix = -hamming_dist
        select_fn = torch.topk
    else:
        raise ValueError(f"Unsporting: {distance}，select 'cosine' 或 'hamming'")

    pseudo_aligned_data = []

    for i in range(len(unaligned_data)):
        img_emb = image_embeddings[i]
        category = categories[i]
        sim_scores = sim_matrix[i]  # [N]

        # get most similar top-k texts
        topk_values, topk_indices = select_fn(sim_scores, top_k)
        topk_texts = text_embeddings[topk_indices]

        if weighted:
            # for Hamming distance, we convert to similarity scores
            if distance.lower() == 'hamming':
                # transform to similarity scores
                similarity_scores = topk_values.max() - topk_values + 1
            else:
                similarity_scores = topk_values

            # apply softmax weights
            weights = F.softmax(similarity_scores, dim=0).unsqueeze(1)  # [k, 1]
            fused_text_emb = (topk_texts * weights).sum(dim=0)
        else:
            fused_text_emb = topk_texts.mean(dim=0)

        pseudo_aligned_data.append({
            'image_embedding': img_emb,
            'text_embedding': fused_text_emb,
            'category': category,
            'is_aligned': 1
        })

    return pseudo_aligned_data


def build_pseudo_aligned_IEMOCAP(unaligned_data, topk=5, distance='Euclidean', device='cuda:1', epsilon = 0.1):
    """
    identify top-k most similar text embeddings for each image embedding
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    video_embeddings = torch.stack([d['video_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)
    audio_embeddings = torch.stack([d['audio_embedding'] for d in unaligned_data]).to(device)
    text_topk_indices_list, udio_topk_indices_list = [], []
    if distance == 'Euclidean':
        text_topk_indices_list = compute_topk_Euclidean_indices(video_embeddings, txt_embeddings, topk=topk)
        audio_topk_indices_list = compute_topk_Euclidean_indices(video_embeddings, audio_embeddings, topk=topk)
    elif distance == 'Hamming':
        text_topk_indices_list = compute_topk_Hamming_indices(video_embeddings, txt_embeddings, topk=topk)
        audio_topk_indices_list = compute_topk_Hamming_indices(video_embeddings, audio_embeddings, topk=topk)
    elif distance == 'PrivateHamming':
        text_topk_indices_list = compute_private_hamming_topk_indices(
            img_embeddings=video_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, r=10, seed=42)
        audio_topk_indices_list = compute_private_hamming_topk_indices(
            img_embeddings=video_embeddings, txt_embeddings=audio_embeddings,
            k=topk, chunk_size=2048, epsilon=epsilon, r=10, seed=42)
    elif distance == 'TPOneHot':
        text_topk_indices_list = compute_TPOneHot_topk_indices(
            img_embeddings=video_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True)
        audio_topk_indices_list = compute_TPOneHot_topk_indices(
            img_embeddings=video_embeddings, txt_embeddings=audio_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    # construct new dataset
    new_dataset = []
    for i, topk_indices in enumerate(text_topk_indices_list):
        video_emb = video_embeddings[i]
        topk_audio_emb = audio_embeddings[audio_topk_indices_list[i]]
        topk_txt_emb = txt_embeddings[topk_indices]  # [topk, D]

        new_dataset.append({
            'video_embedding': video_emb.cpu(),
            'audio_embedding':  topk_audio_emb.cpu(),
            'text_embedding': topk_txt_emb.cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1)
        })

    return new_dataset



def build_pseudo_aligned_dataset(unaligned_data, topk=5, distance='Euclidean', device='cuda', epsilon = 0.1):
    """
    identify top-k most similar text embeddings for each image embedding
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    img_embeddings = torch.stack([d['image_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)
    topk_indices_list = []
    if distance == 'Euclidean':
        topk_indices_list = compute_topk_Euclidean_indices(img_embeddings, txt_embeddings, topk=topk)
    elif distance == 'Hamming':
        topk_indices_list = compute_topk_Hamming_indices(img_embeddings, txt_embeddings, topk=topk)
    elif distance == 'PrivateHamming':
        topk_indices_list = compute_private_hamming_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, r=10, seed=42)
    elif distance == 'BitFlipping':
        topk_indices_list = compute_bit_flipping_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, orthogonal=True)
    elif distance == 'TPOneHot':
        topk_indices_list = compute_TPOneHot_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    # construct new dataset
    new_dataset = []
    for i, topk_indices in enumerate(topk_indices_list):
        image_emb = img_embeddings[i]
        topk_txt_emb = txt_embeddings[topk_indices]  # [topk, D]

        new_dataset.append({
            'image_embedding': image_emb.cpu(),
            'text_embedding': topk_txt_emb.cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1)
        })

    return new_dataset




def build_pseudo_aligned_dataset_t2i(unaligned_data, topk=5, distance='Euclidean', device='cuda', epsilon=0.1):
    """
    identify top-k most similar image embeddings for each text embedding (text->image)
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' (top-k, D) and 'text_embedding' (D)
    """
    img_embeddings = torch.stack([d['image_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)

    if distance == 'Euclidean':
        topk_indices_list = compute_topk_Euclidean_indices(txt_embeddings, img_embeddings, topk=topk)
    elif distance == 'Hamming':
        topk_indices_list = compute_topk_Hamming_indices(txt_embeddings, img_embeddings, topk=topk)
    elif distance == 'PrivateHamming':
        topk_indices_list = compute_private_hamming_topk_indices(
            img_embeddings=txt_embeddings, txt_embeddings=img_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, r=10, seed=42)
    elif distance == 'BitFlipping':
        topk_indices_list = compute_bit_flipping_topk_indices(
            img_embeddings=txt_embeddings, txt_embeddings=img_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, orthogonal=True)
    elif distance == 'TPOneHot':
        topk_indices_list = compute_TPOneHot_topk_indices(
            img_embeddings=txt_embeddings, txt_embeddings=img_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    new_dataset = []
    for i, topk_indices in enumerate(topk_indices_list):
        text_emb = txt_embeddings[i]
        topk_img_emb = img_embeddings[topk_indices]  # [topk, D]

        new_dataset.append({
            'image_embedding': topk_img_emb.cpu(),
            'text_embedding': text_emb.cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1)
        })

    return new_dataset



def build_missing_pseudo_aligned_dataset(img_embeds, txt_embeds, topk=5,
                                         distance='Euclidean', device='cuda:1', epsilon = 0.1):
    """
    identify top-k most similar text embeddings for each image embedding
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    img_embeddings = img_embeds.to(device)
    txt_embeddings = txt_embeds.to(device)
    topk_indices_list = []
    if distance == 'Euclidean':
        topk_indices_list = compute_topk_Euclidean_indices(img_embeddings, txt_embeddings, topk=topk)
    elif distance == 'Hamming':
        topk_indices_list = compute_topk_Hamming_indices(img_embeddings, txt_embeddings, topk=topk)
    elif distance == 'PrivateHamming':
        topk_indices_list = compute_private_hamming_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, r=10, seed=42)
    elif distance == 'BitFlipping':
        topk_indices_list = compute_bit_flipping_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, orthogonal=True)
    elif distance == 'TPOneHot':
        topk_indices_list = compute_TPOneHot_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    # construct new dataset
    new_dataset = []
    for i, topk_indices in enumerate(topk_indices_list):
        print(i, topk_indices)
        image_emb = img_embeddings[i]
        topk_txt_emb = txt_embeddings[topk_indices]  # [topk, D]

        new_dataset.append({
            'image_embedding': image_emb.cpu(),
            'text_embedding': topk_txt_emb.cpu(),
            'category': torch.tensor(-1)
        })

    return new_dataset



def compute_topk_Euclidean_indices(img_embeddings, txt_embeddings, topk=5, chunk_size=512):
    """
    compute top-k most similar text indices for each image embedding based on Euclidean distance.
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    img_embeddings = F.normalize(img_embeddings, dim=1)
    txt_embeddings = F.normalize(txt_embeddings, dim=1)

    topk_indices_list = []

    for start in range(0, img_embeddings.size(0), chunk_size):
        end = min(start + chunk_size, img_embeddings.size(0))
        img_chunk = img_embeddings[start:end]  # [chunk_size, D]

        # similarity [chunk_size, N_txt]
        sim_chunk = img_chunk @ txt_embeddings.T

        # top-k
        _, topk_indices = torch.topk(sim_chunk, k=topk, dim=1)
        topk_indices_list.extend(topk_indices.cpu().tolist())

    return topk_indices_list


def sign_hashing_binary(embeddings):
    """
    encode embeddings to binary vectors using sign hashing.
    Args:
        embeddings (torch.Tensor): [N, D]
    Returns:
        torch.Tensor: [N, D] 0/1
    """
    return (embeddings > 0).int()


def compute_topk_Hamming_indices(img_embeddings, txt_embeddings, topk=5, chunk_size=64):
    """
    compute top-k most similar text indices for each image embedding based on Hamming distance.
    Args:
        img_bin: [N_img, D] （0/1）
        txt_bin: [N_txt, D] （0/1）
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    img_bin = sign_hashing_binary(img_embeddings)
    txt_bin = sign_hashing_binary(txt_embeddings)
    img_bin, txt_bin = generate_lsh_embeddings(img_embeddings, txt_embeddings)
    topk_indices_list = []

    for start in range(0, img_bin.size(0), chunk_size):
        end = min(start + chunk_size, img_bin.size(0))
        img_chunk = img_bin[start:end]  # [chunk, D]

        # xor operation to compute Hamming distance
        # Expand: [chunk, 1, D] vs. [1, N_txt, D] => [chunk, N_txt, D]
        xor = (img_chunk.unsqueeze(1) != txt_bin.unsqueeze(0))  # bool
        hamming_dist = xor.sum(dim=2).to(torch.float32)  # [chunk, N_txt]

        # top-k
        _, topk_indices = torch.topk(-hamming_dist, k=topk, dim=1)
        topk_indices_list.extend(topk_indices.cpu().tolist())

    return topk_indices_list

def compute_private_hamming_topk_indices(img_embeddings, txt_embeddings, k=5, chunk_size=512, epsilon=0.1, r=10, seed=42, orthogonal=True):
    img_embeddings = normalize_embeddings(img_embeddings)
    txt_embeddings = normalize_embeddings(txt_embeddings)
    shift = estimate_shift(img_embeddings, txt_embeddings)  # [D]
    if orthogonal:
        lsh_projections = generate_orthogonal_lsh_projections(shift, dim=img_embeddings.size(1), num_vecs=512)
    else:
        lsh_projections = generate_random_lsh_vectors(dim=img_embeddings.size(1), num_vecs=512)
    img_bins = lsh_hash_bits(img_embeddings, lsh_projections)
    txt_bins = lsh_hash_bits(txt_embeddings, lsh_projections)
    # initialize
    alpha = 2 * epsilon * (1 - epsilon)
    phi = 2 * r
    # Generate disjoint hashes
    m = int(r*1024)
    H0, H1 = generate_disjoint_hashes(n=img_bins.size(1), r=r, m=m, seed=seed)

    # Encode image and text binary vectors
    img_encoded = encode_matrix_torch(img_bins, H0, H1, m)
    txt_encoded = encode_matrix_torch(txt_bins, H0, H1, m)

    img_flipped = bit_flip_matrix_torch(img_encoded,epsilon)
    txt_flipped = bit_flip_matrix_torch(txt_encoded,epsilon)

    N_img = img_flipped.shape[0]
    topk_indices = []

    for start in tqdm(range(0, N_img, chunk_size), desc='Computing top-k indices'):
        end= min(start + chunk_size, N_img)
        img_chunk = img_flipped[start:end]
        corrected_hamming_dist = corrected_hamming_distance_chunked(img_chunk,txt_flipped,alpha, phi, m)  # [chunk_size, N_txt]
        # topk-k
        _, topk_indices_chunk = torch.topk(-corrected_hamming_dist, k=k, dim=1)
        topk_indices_chunk = topk_indices_chunk.cpu().tolist()
        topk_indices.extend(topk_indices_chunk)

    return topk_indices


def compute_bit_flipping_topk_indices(img_embeddings, txt_embeddings, k=5, chunk_size=16, epsilon=0.1, orthogonal=True):
    img_embeddings = normalize_embeddings(img_embeddings)
    txt_embeddings = normalize_embeddings(txt_embeddings)
    shift = estimate_shift(img_embeddings, txt_embeddings)
    proj = generate_orthogonal_lsh_projections(shift, img_embeddings.size(1), 512) if orthogonal else \
           generate_random_lsh_vectors(img_embeddings.size(1), 512)

    img_bins = lsh_hash_bits(img_embeddings, proj)
    txt_bins = lsh_hash_bits(txt_embeddings, proj)

    img_flipped = bit_flip_matrix_torch(img_bins, epsilon)
    txt_flipped = bit_flip_matrix_torch(txt_bins, epsilon)

    topk_indices = []

    for i in tqdm(range(img_flipped.size(0)), desc='Computing top-k indices'):
        img_vec = img_flipped[i]  # [num_bits]
        # Compute Hamming distance
        dist = (img_vec != txt_flipped).sum(dim=1).to(torch.float32)  # [N_txt]
        topk = torch.topk(-dist, k=k, dim=0).indices.cpu().tolist()
        topk_indices.append(topk)

    return topk_indices

def compute_TPOneHot_topk_indices(img_embeddings, txt_embeddings, k=5, chunk_size=512, epsilon=0.1, seed=42, orthogonal=True):
    img_embeddings = normalize_embeddings(img_embeddings)
    txt_embeddings = normalize_embeddings(txt_embeddings)
    shift = estimate_shift(img_embeddings, txt_embeddings)  # [D]
    if orthogonal:
        lsh_projections = generate_orthogonal_lsh_projections(shift, dim=img_embeddings.size(1), num_vecs=512)
    else:
        lsh_projections = generate_random_lsh_vectors(dim=img_embeddings.size(1), num_vecs=512)
    img_bins = lsh_hash_bits(img_embeddings, lsh_projections)
    txt_bins = lsh_hash_bits(txt_embeddings, lsh_projections)
    # initialize

    # # Encode image and text binary vectors
    H0, H1, m = generate_tpoh_hashes(n=img_bins.size(1))
    img_encoded = encode_tpoh_torch(img_bins, H0, H1, m)
    txt_encoded = encode_tpoh_torch(txt_bins, H0, H1, m)

    img_flipped = bit_flip_matrix_torch(img_encoded,epsilon)
    txt_flipped = bit_flip_matrix_torch(txt_encoded,epsilon)

    print(img_flipped.shape)

    N_img = img_flipped.shape[0]
    topk_indices = []

    for start in tqdm(range(0, N_img, chunk_size), desc='Computing top-k indices'):
        end= min(start + chunk_size, N_img)
        img_chunk = img_flipped[start:end]
        raw = compute_hamming_distance_chunked(img_chunk, txt_flipped)
        # topk-k
        _, topk_indices_chunk = torch.topk(-raw, k=k, dim=1)
        topk_indices_chunk = topk_indices_chunk.cpu().tolist()
        topk_indices.extend(topk_indices_chunk)

    return topk_indices

def compute_bidirectional_mappings(img_embeddings, txt_embeddings,
                                   topk_text=5, topk_image=5,
                                   distance='PrivateHamming', device='cuda', **kwargs):
    """
    返回两个 list-of-lists：
      - topk_texts_per_image: length N_img, each is list of topk_text indices (K_t)
      - topk_images_per_text:  length N_txt, each is list of topk_image indices (K_i)
    img_embeddings, txt_embeddings: torch tensors on device
    """
    # 1) image -> text (你已有)
    if distance == 'PrivateHamming':
        topk_texts_per_image = compute_private_hamming_topk_indices(
            img_embeddings, txt_embeddings, k=topk_text, **kwargs)
    elif distance == 'TPOneHot':
        topk_texts_per_image = compute_TPOneHot_topk_indices(
            img_embeddings, txt_embeddings, k=topk_text, **kwargs)
    else:
        # fallback to Euclidean
        topk_texts_per_image = compute_topk_Euclidean_indices(img_embeddings, txt_embeddings, topk=topk_text)

    # 2) text -> image (swap queries and candidates)
    if distance == 'PrivateHamming':
        topk_images_per_text = compute_private_hamming_topk_indices(
            txt_embeddings, img_embeddings, k=topk_image, **kwargs)
    elif distance == 'TPOneHot':
        topk_images_per_text = compute_TPOneHot_topk_indices(
            txt_embeddings, img_embeddings, k=topk_image, **kwargs)
    else:
        topk_images_per_text = compute_topk_Euclidean_indices(txt_embeddings, img_embeddings, topk=topk_image)

    return topk_texts_per_image, topk_images_per_text


def compute_similarity_matrix(img_embeddings, txt_embeddings, chunk_size=512):
    """
    compute similarity matrix between image and text embeddings.
    :return: similarity matrix
    """
    img_embeddings = F.normalize(img_embeddings, dim=1)
    txt_embeddings = F.normalize(txt_embeddings, dim=1)

    N_img = img_embeddings.size(0)
    N_txt = txt_embeddings.size(0)
    sim_matrix = torch.zeros(N_img, N_txt, device=img_embeddings.device)

    for start in range(0, N_img, chunk_size):
        end = min(start + chunk_size, N_img)
        sim_chunk = img_embeddings[start:end] @ txt_embeddings.T  # [chunk_size, N_txt]
        sim_matrix[start:end] = sim_chunk

    return sim_matrix

class PseudoAlignedDataset(Dataset):
    def __init__(self, image_embeddings, topk_text_embeddings,categories):
        self.image_embeddings = image_embeddings
        self.topk_text_embeddings = topk_text_embeddings
        self.categories = categories

    def __len__(self):
        return self.image_embeddings.shape[0]

    def __getitem__(self, idx):
        return {
            'image_embedding': self.image_embeddings[idx],            # [D]
            'text_embedding': self.topk_text_embeddings[idx],    # [k, D]
             'category': self.categories[idx]
        }

def build_pseudo_aligned_dataset_from_aligned(aligned_dataset, k=5, device='cuda'):
    """
    args: aligned_dataset（
    :return PseudoAlignedDataset
    """

    image_embeddings = []
    text_embeddings = []
    categories = []

    for item in aligned_dataset:
        image_embeddings.append(item['image_embedding'])  # Tensor [D]
        text_embeddings.append(item['text_embedding'])    # Tensor [D]
        categories.append(item.get('category', torch.tensor(-1)))
    image_embeddings = torch.stack(image_embeddings).to(device)  # [N, D]
    text_embeddings = torch.stack(text_embeddings).to(device)    # [N, D]
    categories = torch.stack(categories).to(device)  # [N]
    # categories = torch.tensor(categories, device=device)

    # repeat text embeddings to create top-k
    topk_text_embeddings = text_embeddings.unsqueeze(1).repeat(1, k, 1)  # [N, k, D]

    return PseudoAlignedDataset(image_embeddings, topk_text_embeddings, categories)




if __name__ == "__main__":
    # Example usage
    from data_loader.MSCOCODataset import CocoClipDataset, clip_collate_fn
    from data_loader.Flickr30kDataset import Flickr30kDataset, flickr_collate_fn
    from data_loader.IEMOCAPDataset import IEMOCAPDataset, IEMOCAP_collate_fn
    from encoder.CLIPEncoder import CLIPEncoder
    from encoder.ImageBindEncoder import ImageBindEncoder
    from torchvision import transforms
    from transformers import AutoTokenizer


    #clip_encoder = CLIPEncoder(device='cuda')
    imagebind_encoder = ImageBindEncoder(device='cuda:1')
    # img_dir = './MSCOCO/val2017'
    # caption_ann_file = './MSCOCO/annotations/captions_val2017.json'
    # category_ann_file = './MSCOCO/annotations/instances_val2017.json'
    # dataset = CocoClipDataset(
    #     img_dir=img_dir ,
    #     caption_ann_file=caption_ann_file,
    #     category_ann_file=category_ann_file
    # )

    # img_dir = './data/dataset/Flicker30k/Images/'
    # caption_file = './data/dataset/Flicker30k/captions.txt'

    # Create dataset with image validation
    # dataset = Flickr30kDataset(
    #     img_dir=img_dir,
    #     caption_file=caption_file,
    #     caption_strategy='first',
    #     validate_images=False  # Enable full validation
    # )
    #
    # save_path = './pretrained_embeddings/Flicker30k/imagebind_embeddings.pt'
    # save_Flicker30k_imagebind_embeddings(dataset, imagebind_encoder, save_path, batch_size=512, collate_fn=flickr_collate_fn)
    # save_path = './pretrained_embeddings/Flicker30k/clip_embeddings.pt'

    # IEMOCAP
    dataset = IEMOCAPDataset(data_path='/data2/kudret/data/dataset/IEMOCAP/test',
                             txt_path='/data2/kudret/data/dataset/IEMOCAP/test.txt')
    save_path = '/data2/kudret/codes/FedML/pretrained_embeddings/IEMOCAP/imagebind_embeddings_test.pt'
    save_IEMOCAP_imagebind_embeddings(dataset, imagebind_encoder, save_path, batch_size=64, collate_fn=IEMOCAP_collate_fn)