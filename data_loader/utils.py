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
try:
    from imagebind.models.imagebind_model import ModalityType
except ModuleNotFoundError:
    # imagebind is only needed for embedding export helpers; core training code does not depend on it
    ModalityType = None
from trainer.PrivateHamming import (encode_matrix, bit_flip_matrix, corrected_hamming_matrix, generate_disjoint_hashes,
                                    encode_matrix_torch,encode_matrix_chunked, bit_flip_matrix_torch,
                                    corrected_hamming_matrix_torch, corrected_hamming_distance_chunked)
from trainer.OrthogonalProjection import (generate_orthogonal_lsh_projections,estimate_shift,
                                          lsh_hash_bits, normalize_embeddings, generate_random_lsh_vectors,
                                          generate_lsh_embeddings)
from trainer.TPOneHot import (generate_tpoh_hashes,encode_tpoh_torch,compute_hamming_distance_chunked)
from trainer.Coreset import apply_coreset_after_topk


def print_hashcoreset_diagnostics(tag, diagnostics):
    print(
        f"[HashCoreset {tag} diagnostics] "
        f"queries={diagnostics['num_queries']} reps={diagnostics['num_reps']} "
        f"compression={diagnostics['compression_ratio']:.3f} nonrep={diagnostics['nonrep_ratio']:.3f}"
    )
    print(
        f"[HashCoreset {tag} diagnostics] "
        f"buckets={diagnostics['num_buckets']} "
        f"bucket_size(mean/median/p90/max)="
        f"{diagnostics['bucket_size_mean']:.2f}/"
        f"{diagnostics['bucket_size_median']:.2f}/"
        f"{diagnostics['bucket_size_p90']:.2f}/"
        f"{diagnostics['bucket_size_max']:.0f}"
    )
    print(
        f"[HashCoreset {tag} diagnostics] "
        f"gt_top1(before->after)={diagnostics['orig_top1_is_gt_rate']:.4f}->"
        f"{diagnostics['new_top1_is_gt_rate']:.4f}, "
        f"gt_in_topk(before->after)={diagnostics['gt_in_orig_topk_rate']:.4f}->"
        f"{diagnostics['gt_in_new_topk_rate']:.4f}"
    )
    print(
        f"[HashCoreset {tag} diagnostics] "
        f"top1_preserved={diagnostics['top1_preserved_rate']:.4f}, "
        f"orig_top1_kept_in_new_topk={diagnostics['orig_top1_kept_in_new_topk_rate']:.4f}, "
        f"top1_agrees_with_rep={diagnostics['query_top1_agrees_with_rep_rate']:.4f}, "
        f"unique_new_top1_ratio={diagnostics['unique_new_top1_ratio']:.4f}"
    )


def reciprocal_rerank_topk(
    forward_topk,
    backward_topk,
    *,
    forward_topk_distances=None,
    backward_topk_distances=None,
    topk_out=None,
    incoming_cap=None,
    row_weight=1.0,
    reciprocal_weight=1.0,
    col_weight=1.0,
    top1_weight=0.3,
    forward_distance_weight=1.0,
    backward_distance_weight=1.0,
    distance_tau=8.0,
    tag="i2t",
):
    """
    Reciprocal rerank on a bipartite top-k graph.

    `forward_topk` is the candidate list we want to rerank. `backward_topk`
    provides reciprocal evidence from the opposite retrieval direction.

    When distance tables are provided, the distance scoring follows the
    bidirectional matching probability idea from cross-modal learning:

        p_fwd(q, c) = exp(-d_fwd(q,c) / tau) / sum_k exp(-d_fwd(q,k) / tau)
        p_bwd(c, q) = exp(-d_bwd(c,q) / tau) / sum_k exp(-d_bwd(c,k) / tau)

    These softmax probabilities are more principled than rank-based or
    linear-normalized scores: they are locally comparable within each query's
    candidate set and reflect actual distance magnitudes.

    `distance_tau` controls the temperature of the softmax.  A smaller tau
    sharpens the distribution (first-ranked dominates); a larger tau flattens
    it (ranks matter less than reciprocal evidence).
    """
    def _softmax_distance_scores(distance_map, tau):
        """Convert a {idx: hamming_distance} map to softmax probability scores.
        Lower distance → higher probability score."""
        if not distance_map:
            return {}
        indices = list(distance_map.keys())
        dists = torch.tensor([float(distance_map[i]) for i in indices], dtype=torch.float32)
        probs = torch.softmax(-dists / max(float(tau), 1e-6), dim=0)
        return {idx: float(p) for idx, p in zip(indices, probs)}

    if topk_out is None:
        topk_out = len(forward_topk[0]) if forward_topk else 0
    if incoming_cap is None:
        incoming_cap = topk_out

    num_queries = len(forward_topk)
    backward_rank_maps = []
    backward_distance_maps = []
    incoming_candidates = [[] for _ in range(num_queries)]

    for other_idx, row in enumerate(backward_topk):
        rank_map = {}
        dist_map = {}
        row_distances = (
            backward_topk_distances[other_idx]
            if backward_topk_distances is not None and other_idx < len(backward_topk_distances)
            else None
        )
        for rank, query_idx in enumerate(row):
            query_idx = int(query_idx)
            if query_idx < 0 or query_idx in rank_map:
                continue
            rank_map[query_idx] = rank
            if row_distances is not None and rank < len(row_distances):
                dist_map[query_idx] = float(row_distances[rank])
            if 0 <= query_idx < num_queries:
                incoming_candidates[query_idx].append((other_idx, rank))
        backward_rank_maps.append(rank_map)
        backward_distance_maps.append(_softmax_distance_scores(dist_map, distance_tau))

    reranked = []
    total_candidate_pool = 0
    total_added_candidates = 0
    total_reciprocal_final = 0

    for query_idx, row in enumerate(forward_topk):
        forward_rank = {}
        forward_distance_map = {}
        candidate_pool = []
        row_distances = (
            forward_topk_distances[query_idx]
            if forward_topk_distances is not None and query_idx < len(forward_topk_distances)
            else None
        )
        for rank, candidate_idx in enumerate(row):
            candidate_idx = int(candidate_idx)
            if candidate_idx < 0 or candidate_idx in forward_rank:
                continue
            forward_rank[candidate_idx] = rank
            if row_distances is not None and rank < len(row_distances):
                forward_distance_map[candidate_idx] = float(row_distances[rank])
            candidate_pool.append(candidate_idx)
        forward_distance_scores = _softmax_distance_scores(forward_distance_map, distance_tau)

        extras = sorted(incoming_candidates[query_idx], key=lambda item: item[1])
        if incoming_cap is not None:
            extras = extras[:incoming_cap]
        for candidate_idx, _ in extras:
            if candidate_idx not in forward_rank:
                candidate_pool.append(candidate_idx)
                total_added_candidates += 1

        total_candidate_pool += len(candidate_pool)

        scored_candidates = []
        for candidate_idx in candidate_pool:
            row_rank = forward_rank.get(candidate_idx)
            back_rank = None
            row_distance_score = forward_distance_scores.get(candidate_idx, 0.0)
            back_distance_score = 0.0
            if 0 <= candidate_idx < len(backward_rank_maps):
                back_rank = backward_rank_maps[candidate_idx].get(query_idx)
                back_distance_score = backward_distance_maps[candidate_idx].get(query_idx, 0.0)

            reciprocal_flag = back_rank is not None
            row_score = 1.0 / (1.0 + row_rank) if row_rank is not None else 0.0
            col_score = 1.0 / (1.0 + back_rank) if back_rank is not None else 0.0
            top1_bonus = 1.0 if row_rank == 0 and back_rank == 0 else 0.0
            score = (
                row_weight * row_score
                + reciprocal_weight * float(reciprocal_flag)
                + col_weight * col_score
                + top1_weight * top1_bonus
                + forward_distance_weight * row_distance_score
                + backward_distance_weight * back_distance_score
            )
            scored_candidates.append({
                'candidate_idx': candidate_idx,
                'score': score,
                'row_rank': row_rank if row_rank is not None else 10 ** 9,
                'back_rank': back_rank if back_rank is not None else 10 ** 9,
            })

        scored_candidates.sort(
            key=lambda item: (
                -item['score'],
                item['row_rank'],
                item['back_rank'],
                item['candidate_idx'],
            )
        )
        final_row = [item['candidate_idx'] for item in scored_candidates[:topk_out]]
        total_reciprocal_final += sum(
            1 for candidate_idx in final_row
            if 0 <= candidate_idx < len(backward_rank_maps)
            and query_idx in backward_rank_maps[candidate_idx]
        )
        reranked.append(final_row)

    denom = max(num_queries, 1)
    reciprocal_rate = total_reciprocal_final / float(max(num_queries * max(topk_out, 1), 1))
    print(
        f"[ReciprocalRerank {tag}] "
        f"queries={num_queries} avg_candidate_pool={total_candidate_pool / denom:.2f} "
        f"added_candidates={total_added_candidates} final_mutual_rate={reciprocal_rate:.4f}"
    )
    return reranked


def _apply_hashcoreset_pipeline(
    topk_indices_list,
    query_bits,
    *,
    direction_tag,
    coreset_ratio,
    coreset_max_prefix_len,
    coreset_seed,
    prune_to_coreset_reps,
    hashcoreset_diagnostics,
):
    result = apply_coreset_after_topk(
        topk_indices_list,
        query_bits,
        topk_distances=None,
        coreset_ratio=coreset_ratio,
        max_prefix_len=coreset_max_prefix_len,
        seed=coreset_seed,
        return_diagnostics=hashcoreset_diagnostics,
    )
    if hashcoreset_diagnostics:
        topk_indices_list, reps, assign, diagnostics = result
        print_hashcoreset_diagnostics(direction_tag, diagnostics)
    else:
        topk_indices_list, reps, assign = result
    selected_query_indices = reps.cpu().tolist() if prune_to_coreset_reps else None
    print(
        f"[HashCoreset {direction_tag}] coreset_ratio={coreset_ratio} "
        f"reps={int(reps.numel())}/{int(assign.numel())} "
        f"pruned_train_queries={len(selected_query_indices) if selected_query_indices is not None else len(topk_indices_list)}"
    )
    return topk_indices_list, selected_query_indices


def build_bidirectional_pseudo_aligned_datasets_hashcoreset_reciprocal(
    unaligned_data,
    topk=5,
    device='cuda',
    epsilon=0.1,
    *,
    coreset_ratio=0.8,
    coreset_max_prefix_len=20,
    coreset_seed=0,
    prune_to_coreset_reps=False,
    hashcoreset_diagnostics=False,
    reciprocal_incoming_cap=None,
    use_reciprocal_rerank=True,
):
    """
    Scheme A:
      1. build bidirectional TPOneHot top-k lists
      2. (optional) reciprocal rerank each direction with the opposite direction
      3. apply HashCoreset on the reranked lists
      4. materialize i2t / t2i pseudo-aligned datasets

    Set `use_reciprocal_rerank=False` to skip step 2 and pass the raw top-k
    lists directly into HashCoreset.  The reciprocal_rerank_topk code is kept
    intact and can be re-enabled by flipping the flag back to True.
    """
    img_embeddings = torch.stack([d['image_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)

    i2t_topk_raw, t2i_topk_raw, i2t_query_bits, t2i_query_bits, i2t_topk_distances, t2i_topk_distances = compute_TPOneHot_bidirectional_topk_indices(
        img_embeddings=img_embeddings,
        txt_embeddings=txt_embeddings,
        k=topk,
        chunk_size=2048,
        epsilon=epsilon,
        seed=42,
        orthogonal=True,
        return_query_bins=True,
        return_topk_distances=True,
    )

    if use_reciprocal_rerank:
        i2t_topk_reranked = reciprocal_rerank_topk(
            i2t_topk_raw,
            t2i_topk_raw,
            forward_topk_distances=i2t_topk_distances,
            backward_topk_distances=t2i_topk_distances,
            topk_out=topk,
            incoming_cap=reciprocal_incoming_cap,
            tag="i2t",
        )
        t2i_topk_reranked = reciprocal_rerank_topk(
            t2i_topk_raw,
            i2t_topk_raw,
            forward_topk_distances=t2i_topk_distances,
            backward_topk_distances=i2t_topk_distances,
            topk_out=topk,
            incoming_cap=reciprocal_incoming_cap,
            tag="t2i",
        )
    else:
        print("[ReciprocalRerank] skipped (use_reciprocal_rerank=False)")
        i2t_topk_reranked = i2t_topk_raw
        t2i_topk_reranked = t2i_topk_raw

    i2t_topk_final, selected_i2t_indices = _apply_hashcoreset_pipeline(
        i2t_topk_reranked,
        i2t_query_bits,
        direction_tag="i2t",
        coreset_ratio=coreset_ratio,
        coreset_max_prefix_len=coreset_max_prefix_len,
        coreset_seed=coreset_seed,
        prune_to_coreset_reps=prune_to_coreset_reps,
        hashcoreset_diagnostics=hashcoreset_diagnostics,
    )
    t2i_topk_final, selected_t2i_indices = _apply_hashcoreset_pipeline(
        t2i_topk_reranked,
        t2i_query_bits,
        direction_tag="t2i",
        coreset_ratio=coreset_ratio,
        coreset_max_prefix_len=coreset_max_prefix_len,
        coreset_seed=coreset_seed,
        prune_to_coreset_reps=prune_to_coreset_reps,
        hashcoreset_diagnostics=hashcoreset_diagnostics,
    )

    dataset_i2t = []
    query_indices_i2t = selected_i2t_indices if selected_i2t_indices is not None else range(len(i2t_topk_final))
    for i in query_indices_i2t:
        topk_indices = i2t_topk_final[i]
        dataset_i2t.append({
            'image_embedding': img_embeddings[i].cpu(),
            'text_embedding': txt_embeddings[topk_indices].cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1),
        })

    dataset_t2i = []
    query_indices_t2i = selected_t2i_indices if selected_t2i_indices is not None else range(len(t2i_topk_final))
    for i in query_indices_t2i:
        topk_indices = t2i_topk_final[i]
        dataset_t2i.append({
            'image_embedding': img_embeddings[topk_indices].cpu(),
            'text_embedding': txt_embeddings[i].cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1),
        })

    return dataset_i2t, dataset_t2i

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



def build_pseudo_aligned_dataset(
    unaligned_data,
    topk=5,
    distance='Euclidean',
    device='cuda',
    epsilon=0.1,
    *,
    coreset_ratio: float = 0.8,
    coreset_max_prefix_len: int = 20,
    coreset_seed: int = 0,
    prune_to_coreset_reps: bool = False,
    hashcoreset_diagnostics: bool = False,
):
    """
    identify top-k most similar text embeddings for each image embedding
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' and 'text_embedding'
    """
    img_embeddings = torch.stack([d['image_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)
    topk_indices_list = []
    selected_query_indices = None
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
    elif distance == 'HashCoreset':
        topk_indices_list, query_bits, topk_distances = compute_TPOneHot_topk_indices(
            img_embeddings=img_embeddings, txt_embeddings=txt_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True,
            return_query_bins=True, return_topk_distances=True)
        result = apply_coreset_after_topk(
            topk_indices_list,
            query_bits,
            topk_distances=topk_distances,
            coreset_ratio=coreset_ratio,
            max_prefix_len=coreset_max_prefix_len,
            seed=coreset_seed,
            return_diagnostics=hashcoreset_diagnostics,
        )
        if hashcoreset_diagnostics:
            topk_indices_list, reps, assign, diagnostics = result
            print_hashcoreset_diagnostics("i2t", diagnostics)
        else:
            topk_indices_list, reps, assign = result
        if prune_to_coreset_reps:
            selected_query_indices = reps.cpu().tolist()
        print(
            f"[HashCoreset i2t] coreset_ratio={coreset_ratio} "
            f"reps={int(reps.numel())}/{int(assign.numel())} "
            f"pruned_train_queries={len(selected_query_indices) if selected_query_indices is not None else len(topk_indices_list)}"
        )
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    # construct new dataset
    new_dataset = []
    query_indices = selected_query_indices if selected_query_indices is not None else range(len(topk_indices_list))
    for i in query_indices:
        topk_indices = topk_indices_list[i]
        image_emb = img_embeddings[i]
        topk_txt_emb = txt_embeddings[topk_indices]  # [topk, D]

        new_dataset.append({
            'image_embedding': image_emb.cpu(),
            'text_embedding': topk_txt_emb.cpu(),
            'category': unaligned_data[i]['category'] if 'category' in unaligned_data[i] else torch.tensor(-1)
        })

    return new_dataset




def build_pseudo_aligned_dataset_t2i(
    unaligned_data,
    topk=5,
    distance='Euclidean',
    device='cuda',
    epsilon=0.1,
    *,
    coreset_ratio: float = 0.6,
    coreset_max_prefix_len: int = 20,
    coreset_seed: int = 0,
    prune_to_coreset_reps: bool = False,
    hashcoreset_diagnostics: bool = False,
):
    """
    identify top-k most similar image embeddings for each text embedding (text->image)
    unaligned_data: list of dicts with 'image_embedding' and 'text_embedding'
    :return: list of dicts with 'image_embedding' (top-k, D) and 'text_embedding' (D)
    """
    img_embeddings = torch.stack([d['image_embedding'] for d in unaligned_data]).to(device)
    txt_embeddings = torch.stack([d['text_embedding'] for d in unaligned_data]).to(device)

    selected_query_indices = None
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
    elif distance == 'HashCoreset':
        topk_indices_list, query_bits, topk_distances = compute_TPOneHot_topk_indices(
            img_embeddings=txt_embeddings, txt_embeddings=img_embeddings, k=topk,
            chunk_size=2048, epsilon=epsilon, seed=42, orthogonal=True,
            return_query_bins=True, return_topk_distances=True)
        result = apply_coreset_after_topk(
            topk_indices_list,
            query_bits,
            topk_distances=topk_distances,
            coreset_ratio=coreset_ratio,
            max_prefix_len=coreset_max_prefix_len,
            seed=coreset_seed,
            return_diagnostics=hashcoreset_diagnostics,
        )
        if hashcoreset_diagnostics:
            topk_indices_list, reps, assign, diagnostics = result
            print_hashcoreset_diagnostics("t2i", diagnostics)
        else:
            topk_indices_list, reps, assign = result
        if prune_to_coreset_reps:
            selected_query_indices = reps.cpu().tolist()
        print(
            f"[HashCoreset t2i] coreset_ratio={coreset_ratio} "
            f"reps={int(reps.numel())}/{int(assign.numel())} "
            f"pruned_train_queries={len(selected_query_indices) if selected_query_indices is not None else len(topk_indices_list)}"
        )
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    new_dataset = []
    query_indices = selected_query_indices if selected_query_indices is not None else range(len(topk_indices_list))
    for i in query_indices:
        topk_indices = topk_indices_list[i]
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

def compute_private_hamming_topk_indices(
    img_embeddings,
    txt_embeddings,
    k=5,
    chunk_size=512,
    epsilon=0.1,
    r=10,
    seed=42,
    orthogonal=True,
    *,
    return_query_bins: bool = False,
):
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

    if return_query_bins:
        return topk_indices, img_bins
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

def compute_TPOneHot_topk_indices(
    img_embeddings,
    txt_embeddings,
    k=5,
    chunk_size=512,
    epsilon=0.1,
    seed=42,
    orthogonal=True,
    *,
    return_query_bins: bool = False,
    return_topk_distances: bool = False,
):
    """
    When `return_query_bins=True`, returns the query-side binary representation used by
    HashCoreset bucketing. We return the TPOneHot-side bit-flipped codes instead of the
    original LSH bits so that coreset grouping matches the retrieval space.
    """
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
    topk_distances = []

    for start in tqdm(range(0, N_img, chunk_size), desc='Computing top-k indices'):
        end= min(start + chunk_size, N_img)
        img_chunk = img_flipped[start:end]
        raw = compute_hamming_distance_chunked(img_chunk, txt_flipped)
        # topk-k
        topk_negdist_chunk, topk_indices_chunk = torch.topk(-raw, k=k, dim=1)
        topk_indices_chunk = topk_indices_chunk.cpu().tolist()
        topk_distances_chunk = (-topk_negdist_chunk).to(torch.long).cpu().tolist()
        topk_indices.extend(topk_indices_chunk)
        topk_distances.extend(topk_distances_chunk)

    if return_query_bins and return_topk_distances:
        return topk_indices, img_flipped.to(torch.bool), topk_distances
    if return_query_bins:
        return topk_indices, img_flipped.to(torch.bool)
    if return_topk_distances:
        return topk_indices, topk_distances
    return topk_indices


def compute_TPOneHot_bidirectional_topk_indices(
    img_embeddings,
    txt_embeddings,
    k=5,
    chunk_size=512,
    epsilon=0.1,
    seed=42,
    orthogonal=True,
    *,
    return_query_bins: bool = False,
    return_topk_distances: bool = False,
):
    """
    Compute i2t and t2i top-k tables in one pass over the chunked distance matrix.

    The expensive Hamming distance block `raw = dist(img_chunk, txt_all)` is shared:
    - row top-k gives image -> text candidates
    - column top-k is updated incrementally to give text -> image candidates
    """
    img_embeddings = normalize_embeddings(img_embeddings)
    txt_embeddings = normalize_embeddings(txt_embeddings)
    shift = estimate_shift(img_embeddings, txt_embeddings)
    if orthogonal:
        lsh_projections = generate_orthogonal_lsh_projections(shift, dim=img_embeddings.size(1), num_vecs=512)
    else:
        lsh_projections = generate_random_lsh_vectors(dim=img_embeddings.size(1), num_vecs=512)
    img_bins = lsh_hash_bits(img_embeddings, lsh_projections)
    txt_bins = lsh_hash_bits(txt_embeddings, lsh_projections)

    H0, H1, m = generate_tpoh_hashes(n=img_bins.size(1))
    img_encoded = encode_tpoh_torch(img_bins, H0, H1, m)
    txt_encoded = encode_tpoh_torch(txt_bins, H0, H1, m)

    img_flipped = bit_flip_matrix_torch(img_encoded, epsilon)
    txt_flipped = bit_flip_matrix_torch(txt_encoded, epsilon)

    print(img_flipped.shape)

    device = img_flipped.device
    N_img = img_flipped.shape[0]
    N_txt = txt_flipped.shape[0]

    i2t_topk_indices = []
    i2t_topk_distances = []

    inf_dist = torch.full((N_txt, k), float("inf"), device=device, dtype=torch.float32)
    neg_one_idx = torch.full((N_txt, k), -1, device=device, dtype=torch.long)
    best_t2i_dist = inf_dist
    best_t2i_idx = neg_one_idx

    for start in tqdm(range(0, N_img, chunk_size), desc='Computing bidirectional top-k indices'):
        end = min(start + chunk_size, N_img)
        img_chunk = img_flipped[start:end]
        raw = compute_hamming_distance_chunked(img_chunk, txt_flipped)  # [B, N_txt]

        # image -> text: row-wise top-k
        row_topk_negdist, row_topk_idx = torch.topk(-raw, k=k, dim=1)
        i2t_topk_indices.extend(row_topk_idx.cpu().tolist())
        i2t_topk_distances.extend((-row_topk_negdist).to(torch.long).cpu().tolist())

        # text -> image: maintain column-wise top-k incrementally
        chunk_dist_t = raw.transpose(0, 1)  # [N_txt, B]
        chunk_img_idx = torch.arange(start, end, device=device, dtype=torch.long)
        chunk_img_idx = chunk_img_idx.unsqueeze(0).expand(N_txt, -1)  # [N_txt, B]

        merged_dist = torch.cat([best_t2i_dist, chunk_dist_t], dim=1)
        merged_idx = torch.cat([best_t2i_idx, chunk_img_idx], dim=1)

        best_t2i_negdist, select = torch.topk(-merged_dist, k=k, dim=1)
        best_t2i_dist = -best_t2i_negdist
        best_t2i_idx = torch.gather(merged_idx, 1, select)

    t2i_topk_indices = best_t2i_idx.cpu().tolist()
    t2i_topk_distances = best_t2i_dist.to(torch.long).cpu().tolist()

    if return_query_bins and return_topk_distances:
        return (
            i2t_topk_indices,
            t2i_topk_indices,
            img_flipped.to(torch.bool),
            txt_flipped.to(torch.bool),
            i2t_topk_distances,
            t2i_topk_distances,
        )
    if return_query_bins:
        return (
            i2t_topk_indices,
            t2i_topk_indices,
            img_flipped.to(torch.bool),
            txt_flipped.to(torch.bool),
        )
    if return_topk_distances:
        return i2t_topk_indices, t2i_topk_indices, i2t_topk_distances, t2i_topk_distances
    return i2t_topk_indices, t2i_topk_indices

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