from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
import torch


def invalid_dist_int64(B: int) -> int:
    return int(B) + 1


@torch.no_grad()
def summarize_coreset_diagnostics(
    original_idx_table: torch.Tensor,
    new_idx_table: torch.Tensor,
    reps: torch.Tensor,
    assign: torch.Tensor,
    fill_value: int = -1,
) -> Dict[str, Any]:
    """
    Summarize how much the coreset changed retrieval candidates.
    Assumes diagonal index i is the ground-truth partner for query i.
    """
    if original_idx_table.ndim != 2 or new_idx_table.ndim != 2:
        raise ValueError("Both original_idx_table and new_idx_table must be [N, k].")
    if original_idx_table.shape != new_idx_table.shape:
        raise ValueError("original_idx_table and new_idx_table must share the same shape.")

    N, k = original_idx_table.shape
    targets = torch.arange(N, device=original_idx_table.device, dtype=torch.long)

    bucket_sizes = torch.unique(assign, return_counts=True)[1].to(torch.float32)
    orig_top1 = original_idx_table[:, 0]
    new_top1 = new_idx_table[:, 0]
    valid_orig_top1 = orig_top1 != fill_value
    valid_new_top1 = new_top1 != fill_value

    rep_top1 = orig_top1[assign]
    valid_rep_top1 = rep_top1 != fill_value

    def _safe_mean(mask: torch.Tensor) -> float:
        if mask.numel() == 0:
            return 0.0
        return mask.to(torch.float32).mean().item()

    def _quantile(x: torch.Tensor, q: float) -> float:
        if x.numel() == 0:
            return 0.0
        return torch.quantile(x, q).item()

    gt_before = (original_idx_table == targets[:, None]).any(dim=1)
    gt_after = (new_idx_table == targets[:, None]).any(dim=1)

    top1_preserved_mask = valid_orig_top1 & valid_new_top1
    top1_agree_with_rep_mask = valid_orig_top1 & valid_rep_top1

    diagnostics: Dict[str, Any] = {
        "num_queries": int(N),
        "topk": int(k),
        "num_reps": int(reps.numel()),
        "compression_ratio": float(reps.numel()) / float(max(N, 1)),
        "nonrep_ratio": float(max(N - int(reps.numel()), 0)) / float(max(N, 1)),
        "num_buckets": int(bucket_sizes.numel()),
        "bucket_size_mean": bucket_sizes.mean().item() if bucket_sizes.numel() else 0.0,
        "bucket_size_median": bucket_sizes.median().item() if bucket_sizes.numel() else 0.0,
        "bucket_size_p90": _quantile(bucket_sizes, 0.9),
        "bucket_size_max": bucket_sizes.max().item() if bucket_sizes.numel() else 0.0,
        "orig_top1_is_gt_rate": _safe_mean(orig_top1 == targets),
        "new_top1_is_gt_rate": _safe_mean(new_top1 == targets),
        "gt_in_orig_topk_rate": _safe_mean(gt_before),
        "gt_in_new_topk_rate": _safe_mean(gt_after),
        "top1_preserved_rate": _safe_mean(new_top1[top1_preserved_mask] == orig_top1[top1_preserved_mask]),
        "orig_top1_kept_in_new_topk_rate": _safe_mean((new_idx_table == orig_top1[:, None]).any(dim=1)),
        "query_top1_agrees_with_rep_rate": _safe_mean(orig_top1[top1_agree_with_rep_mask] == rep_top1[top1_agree_with_rep_mask]),
        "unique_new_top1_ratio": float(torch.unique(new_top1[valid_new_top1]).numel()) / float(max(N, 1)),
    }
    return diagnostics


@torch.no_grad()
def _pack_prefix_bits_to_int64(bits: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    Pack first `prefix_len` bits into an int64 key.

    Args:
      bits: [N, B] bool
      prefix_len: <= 63
    Returns:
      keys: [N] int64
    """
    assert bits.dtype == torch.bool
    prefix_len = int(prefix_len)
    if not (1 <= prefix_len <= 63):
        raise ValueError(f"prefix_len must be in [1,63], got {prefix_len}")
    b = bits[:, :prefix_len].to(torch.int64)  # [N,L]
    w = (1 << torch.arange(prefix_len, device=bits.device, dtype=torch.int64)).view(1, -1)
    return (b * w).sum(dim=1)  # [N]


@torch.no_grad()
def build_bucket_coreset(
    bits: torch.Tensor,          # [N,B] bool
    target_size: int,
    seed: int = 0,
    max_prefix_len: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prefix-bucketing coreset selection.

    Returns:
      reps:   [M] long representative indices (global ids, 0..N-1)
      assign: [N] long, assign[i] = representative global id (0..N-1)
    """
    device = bits.device
    N = int(bits.size(0))
    target_size = int(max(1, min(int(target_size), N)))

    # choose prefix_len to get #buckets close to target_size
    best_keys = None
    for L in range(1, int(max_prefix_len) + 1):
        keys = _pack_prefix_bits_to_int64(bits, L)
        nb = int(torch.unique(keys).numel())
        best_keys = keys
        if nb >= target_size:
            break
    keys = best_keys

    uniq_keys, inv = torch.unique(keys, return_inverse=True)  # inv: [N] bucket id
    nb = int(uniq_keys.numel())

    # Pick the first occurrence as the bucket representative. The main quality
    # control now comes from which buckets are kept when we need to downsample.
    reps = torch.full((nb,), -1, device=device, dtype=torch.long)
    order = torch.argsort(inv)
    inv_s = inv[order]
    idx_s = order
    change = torch.ones_like(inv_s, dtype=torch.bool)
    change[1:] = inv_s[1:] != inv_s[:-1]
    first = idx_s[change]
    bids = inv_s[change]
    reps[bids] = first  # [nb]

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Downsample if too many.
    if int(reps.numel()) > target_size:
        perm = torch.randperm(reps.numel(), generator=g, device=device)
        reps = reps[perm[:target_size]]

    # add extras if too few (from largest buckets)
    if int(reps.numel()) < target_size:
        sizes = torch.bincount(inv, minlength=nb)  # [nb]
        big = torch.argsort(sizes, descending=True)
        is_rep = torch.zeros(N, device=device, dtype=torch.bool)
        is_rep[reps] = True
        extra: List[int] = []
        for b in big.tolist():
            if int(reps.numel()) + len(extra) >= target_size:
                break
            members = torch.nonzero(inv == b, as_tuple=False).view(-1)
            cand = members[~is_rep[members]]
            if int(cand.numel()) == 0:
                continue
            pick = cand[torch.randint(0, cand.numel(), (1,), generator=g, device=device)].item()
            extra.append(int(pick))
            is_rep[pick] = True
        if extra:
            reps = torch.cat([reps, torch.tensor(extra, device=device, dtype=torch.long)], dim=0)

    # build assign: bucket -> kept rep if exists else fallback to self
    rep_bucket = inv[reps]  # [M]
    rep_for_bucket = torch.full((nb,), -1, device=device, dtype=torch.long)
    rep_for_bucket[rep_bucket] = reps
    assign = rep_for_bucket[inv]
    miss = assign < 0
    if bool(miss.any()):
        assign[miss] = torch.arange(N, device=device, dtype=torch.long)[miss]

    return reps, assign


@torch.no_grad()
def build_overlap_coreset(
    idx_table: torch.Tensor,      # [N,k] long
    target_size: int,
    seed: int = 0,
    fill_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Faster overlap-lite coreset selection.

    To keep the extra construction cost low, we only:
    1. inspect the first top-3 unique retrieval candidates per query
    2. search neighbors inside the same top-1 bucket
    3. prefer merges with shared>=2 candidates, then fall back to shared>=1

    This keeps the later representative voting stage unchanged while making the
    grouping depend on retrieval overlap instead of hash prefix.
    """
    device = idx_table.device
    N, k = idx_table.shape
    target_size = int(max(1, min(int(target_size), int(N))))
    if target_size >= N:
        reps = torch.arange(N, device=device, dtype=torch.long)
        return reps, reps.clone()

    overlap_topm = min(3, int(k))
    # Keep the existing top-1 coarse bucket, but split very large buckets once
    # more by top-2 so comparisons stay local enough to avoid noisy merges.
    large_bucket_threshold = 4

    # Materialize compact per-query candidate lists and group queries by top-1.
    query_items: List[List[int]] = []
    query_pos: List[Dict[int, int]] = []
    top1_buckets: Dict[int, List[int]] = {}
    candidate_to_queries: Dict[int, List[int]] = {}

    for i in range(N):
        seen = set()
        items: List[int] = []
        pos: Dict[int, int] = {}
        row = idx_table[i].tolist()
        for x in row:
            x = int(x)
            if x < 0 or x == fill_value or x in seen:
                continue
            seen.add(x)
            items.append(x)
            pos[x] = len(items) - 1
            if len(items) >= overlap_topm:
                break

        query_items.append(items)
        query_pos.append(pos)

        for cand in items:
            candidate_to_queries.setdefault(cand, []).append(i)

        if not items:
            continue

        top1_buckets.setdefault(items[0], []).append(i)

    # For normal buckets we still compare inside the full top-1 bucket. For
    # very large buckets, refine them by top-2 before pairwise comparison.
    local_compare_buckets: List[List[int]] = [[] for _ in range(N)]
    for bucket in top1_buckets.values():
        if len(bucket) > large_bucket_threshold:
            top2_subbuckets: Dict[int, List[int]] = {}
            for qid in bucket:
                items = query_items[qid]
                top2_key = int(items[1]) if len(items) > 1 else -1
                top2_subbuckets.setdefault(top2_key, []).append(qid)
            for subbucket in top2_subbuckets.values():
                for qid in subbucket:
                    local_compare_buckets[qid] = subbucket
        else:
            for qid in bucket:
                local_compare_buckets[qid] = bucket

    # Find the best overlap-based representative candidate for each query, but
    # only inside the same top-1 bucket.
    best_rep = list(range(N))
    best_score = [0.0] * N
    best_shared = [0] * N

    for i in range(N):
        items_i = query_items[i]
        if len(items_i) < 2:
            continue

        bucket = local_compare_buckets[i]
        if len(bucket) <= 1 and len(items_i) < 2:
            continue

        cur_best = i
        cur_score = 0.0
        cur_shared = 0
        pos_i = query_pos[i]

        candidate_neighbors = set(bucket)

        # Allow cross-bucket comparisons when the candidate overlap is already
        # strong enough, even if the top-1 retrieved item differs.
        shared_counts: Dict[int, int] = {}
        for cand in items_i:
            for j in candidate_to_queries.get(cand, []):
                if j == i:
                    continue
                shared_counts[j] = shared_counts.get(j, 0) + 1
        for j, shared_cnt in shared_counts.items():
            items_j = query_items[j]
            if not items_j:
                continue
            if items_j[0] == items_i[0]:
                continue
            if shared_cnt >= 2:
                candidate_neighbors.add(j)

        for j in candidate_neighbors:
            if j == i:
                continue
            items_j = query_items[j]
            if not items_j:
                continue

            pos_j = query_pos[j]
            shared = 0
            score = 0.0
            for cand, rank_i in pos_i.items():
                rank_j = pos_j.get(cand)
                if rank_j is None:
                    continue
                shared += 1
                score += (1.0 / float(rank_i + 1)) + (1.0 / float(rank_j + 1))

            if shared == 0:
                continue

            if (
                shared > cur_shared
                or (shared == cur_shared and score > cur_score)
                or (score == cur_score and shared == cur_shared and j < cur_best)
            ):
                cur_best = j
                cur_score = score
                cur_shared = shared

        best_rep[i] = cur_best
        best_score[i] = cur_score
        best_shared[i] = cur_shared

    # Greedily remove the queries with the strongest merge signal.
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    assign_list = list(range(N))
    is_rep = [True] * N
    removed = 0
    need_remove = N - target_size

    def _find_root(x: int) -> int:
        while assign_list[x] != x:
            assign_list[x] = assign_list[assign_list[x]]
            x = assign_list[x]
        return x

    def _merge_pass(min_shared: int) -> None:
        nonlocal removed
        order = list(range(N))
        order.sort(key=lambda i: (-best_shared[i], -best_score[i], i))
        for i in order:
            if removed >= need_remove:
                return
            if best_rep[i] == i or not is_rep[i] or best_shared[i] < min_shared:
                continue
            r = _find_root(best_rep[i])
            if r == i:
                continue
            assign_list[i] = r
            is_rep[i] = False
            removed += 1

    # First keep only high-confidence overlap merges, then allow weaker ones.
    _merge_pass(min_shared=2)
    if removed < need_remove:
        _merge_pass(min_shared=1)

    # Final lightweight fallback: merge inside the same top-1 bucket only.
    if removed < need_remove:
        fallback_order = torch.randperm(N, generator=g).tolist()
        for i in fallback_order:
            if removed >= need_remove or not is_rep[i]:
                continue

            items_i = query_items[i]
            if not items_i:
                continue

            r = -1
            for j in top1_buckets.get(items_i[0], []):
                root_j = _find_root(j)
                if root_j != i and is_rep[root_j]:
                    r = root_j
                    break

            if r < 0:
                continue

            assign_list[i] = r
            is_rep[i] = False
            removed += 1

    # Hard fallback: if top-1 local merges are still insufficient, merge into any
    # other active representative so we always hit the requested coreset size.
    if removed < need_remove:
        fallback_order = torch.randperm(N, generator=g).tolist()
        active_roots = [i for i, keep in enumerate(is_rep) if keep]
        for i in fallback_order:
            if removed >= need_remove or not is_rep[i]:
                continue

            r = -1
            for cand in active_roots:
                root_cand = _find_root(cand)
                if root_cand != i and is_rep[root_cand]:
                    r = root_cand
                    break

            if r < 0:
                continue

            assign_list[i] = r
            is_rep[i] = False
            removed += 1

    # Keep this exact so both retrieval directions can be fused safely.
    if removed != need_remove:
        raise RuntimeError(
            f"build_overlap_coreset failed to reach target size: "
            f"removed={removed}, need_remove={need_remove}, "
            f"target_size={target_size}, N={N}"
        )

    reps = torch.tensor([i for i, keep in enumerate(is_rep) if keep], device=device, dtype=torch.long)
    assign = torch.tensor([_find_root(i) for i in range(N)], device=device, dtype=torch.long)
    return reps, assign


@torch.no_grad()
def vote_topk_for_reps(
    reps: torch.Tensor,                 # [M] long
    assign: torch.Tensor,               # [N] long -> rep global id
    idx_table: torch.Tensor,            # [N,k] long
    dist_table: Optional[torch.Tensor], # [N,k] int64 or None
    topk_out: int,
    fill_value: int = -1,
    B: int = 512,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Aggregate member top-k lists into representative top-k lists via voting.
    Sort rule: weighted_score desc, then count desc, then avg_dist asc.
    The weighted score mixes:
      - frequency: candidates supported by more members are preferred
      - rank weighting: earlier ranks receive larger votes
      - distance bonus: smaller average distance gets a small extra bonus when available
    """
    device = idx_table.device
    N, k = idx_table.shape
    topk_out = int(min(int(topk_out), int(k)))
    inv_dist = invalid_dist_int64(B)

    idx_rep = torch.full((N, topk_out), fill_value, device=device, dtype=torch.long)
    dist_rep = None
    if dist_table is not None:
        dist_rep = torch.full((N, topk_out), inv_dist, device=device, dtype=torch.long)

    # group members by rep using sorting
    order = torch.argsort(assign)
    rep_sorted = assign[order]
    change = torch.ones_like(rep_sorted, dtype=torch.bool)
    change[1:] = rep_sorted[1:] != rep_sorted[:-1]
    starts = torch.nonzero(change, as_tuple=False).view(-1)
    ends = torch.cat([starts[1:], torch.tensor([N], device=device, dtype=torch.long)], dim=0)

    is_rep = torch.zeros((N,), device=device, dtype=torch.bool)
    is_rep[reps] = True

    for st, ed in zip(starts.tolist(), ends.tolist()):
        r = int(rep_sorted[st].item())  # rep global id
        if r < 0 or r >= N or (not bool(is_rep[r])):
            continue
        members = order[st:ed]  # global ids

        cand = idx_table[members]  # [M,k]
        mask = (cand != fill_value)
        cand_flat = cand[mask]  # [Q]
        if int(cand_flat.numel()) == 0:
            continue

        if dist_table is not None:
            vals = dist_table[members]
            vals_flat = vals[mask].to(torch.float32)  # [Q]
        else:
            vals_flat = None

        # Earlier positions should matter more than later positions.
        rank_weights = (1.0 / torch.arange(1, k + 1, device=device, dtype=torch.float32)).view(1, k)
        rank_flat = rank_weights.expand(cand.size(0), -1)[mask]

        uniq, invu, cnt = torch.unique(cand_flat, return_inverse=True, return_counts=True)

        score = cnt.to(torch.float32)

        rank_score = torch.zeros((uniq.numel(),), device=device, dtype=torch.float32)
        rank_score.scatter_add_(0, invu, rank_flat)
        score = score + rank_score

        # tie-break and extra bonus: average distance
        if vals_flat is not None:
            sumd = torch.zeros((uniq.numel(),), device=device, dtype=torch.float32)
            sumd.scatter_add_(0, invu, vals_flat)
            avgd = sumd / cnt.to(torch.float32)
            score = score + 1.0 / (1.0 + avgd)
        else:
            avgd = torch.zeros((uniq.numel(),), device=device, dtype=torch.float32)

        # Stable lexicographic ordering:
        # 1) smaller avg distance
        # 2) larger raw count
        # 3) larger weighted vote score
        o1 = torch.argsort(avgd, descending=False, stable=True)
        uniq1, cnt1, avgd1, score1 = uniq[o1], cnt[o1], avgd[o1], score[o1]
        o2 = torch.argsort(cnt1.to(torch.float32), descending=True, stable=True)
        uniq2, avgd2, score2 = uniq1[o2], avgd1[o2], score1[o2]
        o3 = torch.argsort(score2, descending=True, stable=True)
        uniq3, avgd3 = uniq2[o3], avgd2[o3]

        take = min(topk_out, int(uniq3.numel()))
        idx_rep[r, :take] = uniq3[:take]
        if dist_rep is not None:
            dist_rep[r, :take] = torch.round(avgd3[:take]).to(torch.long)

    return idx_rep, dist_rep


@torch.no_grad()
def apply_coreset_after_topk(
    topk_indices: List[List[int]] | torch.Tensor,  # [N,k]
    query_bits: torch.Tensor,                      # [N,B] (0/1 or bool)
    *,
    topk_distances: Optional[List[List[int]] | torch.Tensor] = None,
    coreset_ratio: float = 0.8,
    max_prefix_len: int = 20,
    seed: int = 0,
    fill_value: int = -1,
    preserve_orig_topm: int = 3,
    return_diagnostics: bool = False,
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor] | Tuple[List[List[int]], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Apply query-side coreset AFTER you have a full top-k table (e.g. from TPOneHot).

    We first group / assign queries using overlap in their original top-k lists,
    then vote member top-k lists into representative top-k lists. Each query
    finally mixes its own closest hits with the representative-voted list. By
    default we preserve the query's original top-4, then fill the remaining
    slots from the representative side.

    Returns:
      new_topk_indices: list-of-lists length N, each length k
      reps:   [M] reps indices
      assign: [N] rep id per query
    """
    if isinstance(topk_indices, torch.Tensor):
        idx_table = topk_indices
    else:
        idx_table = torch.tensor(topk_indices, dtype=torch.long)

    if idx_table.ndim != 2:
        raise ValueError(f"topk_indices must be [N,k], got {tuple(idx_table.shape)}")

    dist_table = None
    if topk_distances is not None:
        if isinstance(topk_distances, torch.Tensor):
            dist_table = topk_distances
        else:
            dist_table = torch.tensor(topk_distances, dtype=torch.long)
        if dist_table.shape != idx_table.shape:
            raise ValueError(
                f"topk_distances must match topk_indices shape, got dist={tuple(dist_table.shape)}, idx={tuple(idx_table.shape)}"
            )

    bits = query_bits
    if bits.dtype != torch.bool:
        bits = bits.to(torch.bool)
    if bits.ndim != 2 or bits.size(0) != idx_table.size(0):
        raise ValueError(f"query_bits must be [N,B] aligned with topk_indices, got bits={tuple(bits.shape)}, idx={tuple(idx_table.shape)}")

    # run on CPU for stability / avoid GPU memory spikes (topk is small, but N can be large)
    idx_table = idx_table.cpu()
    if dist_table is not None:
        dist_table = dist_table.cpu()
    bits = bits.cpu()

    N, k = idx_table.shape
    target_size = max(1, int(float(N) * float(coreset_ratio)))
    preserve_orig_topm = max(0, min(int(preserve_orig_topm), int(k)))

    reps, assign = build_overlap_coreset(
        idx_table,
        target_size=target_size,
        seed=seed,
        fill_value=fill_value,
    )
    idx_rep, _ = vote_topk_for_reps(
        reps=reps,
        assign=assign,
        idx_table=idx_table,
        dist_table=dist_table,
        topk_out=k,
        fill_value=fill_value,
        B=int(bits.size(1)),
    )

    # materialize per-query topk by looking up rep row
    new_topk: List[List[int]] = []
    new_idx_table = torch.full_like(idx_table, fill_value)
    for i in range(N):
        r = int(assign[i].item())
        rep_list = idx_rep[r].tolist()
        rep_list = [int(x) for x in rep_list if int(x) != fill_value and int(x) >= 0]

        orig = idx_table[i].tolist()
        orig_list = [int(x) for x in orig if int(x) >= 0 and int(x) != fill_value]

        # Keep the query's own closest hits first, then backfill from the coreset
        # representative list. For k=5 this becomes: original top-4 + rep top-1.
        final_list: List[int] = []
        seen = set()

        for x in orig_list[:preserve_orig_topm]:
            if x not in seen:
                final_list.append(x)
                seen.add(x)

        for x in rep_list:
            if x not in seen:
                final_list.append(x)
                seen.add(x)
            if len(final_list) >= k:
                break

        # Fill any remaining slots from the rest of the original query list.
        if len(final_list) < k:
            for x in orig_list[preserve_orig_topm:]:
                if x not in seen:
                    final_list.append(x)
                    seen.add(x)
                if len(final_list) >= k:
                    break

        final_list = final_list[:k]
        new_topk.append(final_list)
        if final_list:
            new_idx_table[i, :len(final_list)] = torch.tensor(final_list, dtype=torch.long)

    if return_diagnostics:
        diagnostics = summarize_coreset_diagnostics(
            original_idx_table=idx_table,
            new_idx_table=new_idx_table,
            reps=reps,
            assign=assign,
            fill_value=fill_value,
        )
        return new_topk, reps, assign, diagnostics

    return new_topk, reps, assign

