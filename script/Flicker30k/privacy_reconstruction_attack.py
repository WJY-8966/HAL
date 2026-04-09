import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.OrthogonalProjection import (
    estimate_shift,
    generate_orthogonal_lsh_projections,
    generate_random_lsh_vectors,
    lsh_hash_bits,
    normalize_embeddings,
)
from trainer.PrivateHamming import (
    bit_flip_matrix_torch,
    encode_matrix_torch,
    generate_disjoint_hashes,
)
from trainer.TPOneHot import encode_tpoh_torch, generate_tpoh_hashes
from utils.utils import seed_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct original embeddings from opaque server-observed messages."
    )
    parser.add_argument(
        "--pretrained_file",
        type=str,
        default="/data/wangjiayi/HAL/pretrained_embeddings/Flicker30k/clip_embeddings.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction", type=str, choices=["i2t", "t2i"], default="t2i")
    parser.add_argument(
        "--privacy_scheme",
        type=str,
        choices=["TPOneHot", "HashCoreset", "PrivateHamming"],
        default="TPOneHot",
    )
    parser.add_argument("--orthogonal", action="store_true", default=False)
    parser.add_argument("--num_hash_bits", type=int, default=512)
    parser.add_argument("--flip_epsilon", type=float, default=0.1)
    parser.add_argument("--private_r", type=int, default=5)
    parser.add_argument("--target_train_ratio", type=float, default=0.7)
    parser.add_argument("--decoder_train_ratio", type=float, default=0.1)
    parser.add_argument("--decoder_epochs", type=int, default=50)
    parser.add_argument("--decoder_batch_size", type=int, default=1024)
    parser.add_argument("--decoder_lr", type=float, default=1e-3)
    parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
    parser.add_argument("--cosine_loss_weight", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--decoder_checkpoint", type=str, default="")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/wangjiayi/HAL/attack_outputs",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def split_raw_data(data, args):
    random.shuffle(data)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    total = len(data)
    target_end = int(total * args.target_train_ratio)
    decoder_end = int(total * (args.target_train_ratio + args.decoder_train_ratio))
    if target_end <= 0 or decoder_end <= target_end or decoder_end >= total:
        raise ValueError(
            "Invalid data split. Adjust target_train_ratio / decoder_train_ratio / max_samples."
        )
    return data[:target_end], data[target_end:decoder_end], data[decoder_end:]


def stack_modalities(raw_subset):
    img = torch.stack([item["image_embedding"] for item in raw_subset]).float()
    txt = torch.stack([item["text_embedding"] for item in raw_subset]).float()
    return img, txt


def build_hidden_server_encoder(all_data, args, device):
    img_embeddings, txt_embeddings = stack_modalities(all_data)
    img_embeddings = normalize_embeddings(img_embeddings.to(device))
    txt_embeddings = normalize_embeddings(txt_embeddings.to(device))

    shift = estimate_shift(img_embeddings, txt_embeddings)
    if args.privacy_scheme in {"TPOneHot", "HashCoreset"}:
        # Strictly mirror HAL/data_loader/utils.py::compute_TPOneHot_topk_indices.
        projections = generate_orthogonal_lsh_projections(
            shift,
            dim=img_embeddings.size(1),
            num_vecs=512,
        )
        img_bins = lsh_hash_bits(img_embeddings, projections)
        txt_bins = lsh_hash_bits(txt_embeddings, projections)
        h0, h1, code_dim = generate_tpoh_hashes(n=img_bins.size(1))
        img_encoded = encode_tpoh_torch(img_bins, h0, h1, code_dim)
        txt_encoded = encode_tpoh_torch(txt_bins, h0, h1, code_dim)
        img_flipped = bit_flip_matrix_torch(img_encoded, args.flip_epsilon)
        txt_flipped = bit_flip_matrix_torch(txt_encoded, args.flip_epsilon)
        return {
            "projections": projections,
            "h0": h0,
            "h1": h1,
            "code_dim": int(img_flipped.size(1)),
            "reference_img_flipped_shape": tuple(img_flipped.shape),
            "reference_txt_flipped_shape": tuple(txt_flipped.shape),
        }

    if args.orthogonal:
        projections = generate_orthogonal_lsh_projections(
            shift,
            dim=img_embeddings.size(1),
            num_vecs=args.num_hash_bits,
        )
    else:
        projections = generate_random_lsh_vectors(
            dim=img_embeddings.size(1),
            num_vecs=args.num_hash_bits,
        ).to(device)

    img_bins = lsh_hash_bits(img_embeddings, projections)
    txt_bins = lsh_hash_bits(txt_embeddings, projections)
    code_dim = 2 * img_bins.size(1) * args.private_r
    h0, h1 = generate_disjoint_hashes(
        n=img_bins.size(1),
        r=args.private_r,
        m=code_dim,
        seed=args.seed,
    )
    img_encoded = encode_matrix_torch(img_bins, h0, h1, code_dim)
    txt_encoded = encode_matrix_torch(txt_bins, h0, h1, code_dim)
    img_flipped = bit_flip_matrix_torch(img_encoded, args.flip_epsilon, seed=args.seed)
    txt_flipped = bit_flip_matrix_torch(txt_encoded, args.flip_epsilon, seed=args.seed)
    return {
        "projections": projections,
        "h0": h0,
        "h1": h1,
        "code_dim": int(img_flipped.size(1)),
        "reference_img_flipped_shape": tuple(img_flipped.shape),
        "reference_txt_flipped_shape": tuple(txt_flipped.shape),
    }


def server_encode_messages(embeddings, server_encoder, args, device):
    embeddings = normalize_embeddings(embeddings.to(device))
    bits = lsh_hash_bits(embeddings, server_encoder["projections"])

    if args.privacy_scheme in {"TPOneHot", "HashCoreset"}:
        encoded = encode_tpoh_torch(
            bits,
            server_encoder["h0"],
            server_encoder["h1"],
            server_encoder["code_dim"],
        )
    else:
        encoded = encode_matrix_torch(
            bits,
            server_encoder["h0"],
            server_encoder["h1"],
            server_encoder["code_dim"],
        )

    flipped = bit_flip_matrix_torch(encoded, args.flip_epsilon, seed=args.seed)
    return flipped.float()


def collect_observed_pairs(raw_subset, server_encoder, args, device):
    img_embeddings, txt_embeddings = stack_modalities(raw_subset)
    if args.direction == "i2t":
        source_embeddings = img_embeddings
    else:
        source_embeddings = txt_embeddings

    # The attacker only sees these opaque uploaded messages, not the encoder internals.
    observed_messages = server_encode_messages(source_embeddings, server_encoder, args, device)
    return observed_messages.cpu(), source_embeddings.float().cpu()


class ReconstructionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def standardize_messages(train_x, eval_x_list):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    normalized = [(x - mean) / std for x in eval_x_list]
    return train_x, normalized


def build_tensor_loader(messages, targets, batch_size, shuffle):
    dataset = TensorDataset(messages, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def reconstruction_loss(recon, target, cosine_weight):
    mse = F.mse_loss(recon, target)
    cosine = 1.0 - F.cosine_similarity(recon, target, dim=1).mean()
    return mse + cosine_weight * cosine, mse, cosine


def evaluate_decoder(decoder, data_loader, device):
    decoder.eval()
    mse_sum = 0.0
    cosine_sum = 0.0
    relative_error_sum = 0.0
    count = 0
    with torch.no_grad():
        for messages, targets in data_loader:
            messages = messages.to(device)
            targets = targets.to(device)
            recon = decoder(messages)
            mse_sum += F.mse_loss(recon, targets, reduction="sum").item()
            cosine_sum += F.cosine_similarity(recon, targets, dim=1).sum().item()
            rel_error = (
                (recon - targets).norm(dim=1)
                / targets.norm(dim=1).clamp_min(1e-8)
            )
            relative_error_sum += rel_error.sum().item()
            count += targets.size(0)
    return {
        "mse": mse_sum / max(count, 1),
        "cosine": cosine_sum / max(count, 1),
        "relative_error": relative_error_sum / max(count, 1),
        "num_samples": int(count),
    }


def train_decoder(decoder, train_loader, eval_loader, args, device):
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )
    best_state = None
    best_eval_mse = float("inf")

    for epoch in range(args.decoder_epochs):
        decoder.train()
        train_loss_sum = 0.0
        train_steps = 0
        for messages, targets in train_loader:
            messages = messages.to(device)
            targets = targets.to(device)
            recon = decoder(messages)
            loss, _, _ = reconstruction_loss(recon, targets, args.cosine_loss_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        eval_metrics = evaluate_decoder(decoder, eval_loader, device)
        avg_train_loss = train_loss_sum / max(train_steps, 1)
        print(
            f"[Decoder] epoch={epoch + 1}/{args.decoder_epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"eval_mse={eval_metrics['mse']:.6f} "
            f"eval_cosine={eval_metrics['cosine']:.6f}"
        )

        if eval_metrics["mse"] < best_eval_mse:
            best_eval_mse = eval_metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in decoder.state_dict().items()}

    if best_state is not None:
        decoder.load_state_dict(best_state)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_torch(args.seed)
    device = resolve_device(args.device)

    data = torch.load(args.pretrained_file)
    target_raw, decoder_raw, eval_raw = split_raw_data(data, args)
    print(
        f"Raw split sizes | target={len(target_raw)} "
        f"decoder_train={len(decoder_raw)} eval={len(eval_raw)}"
    )

    server_encoder = build_hidden_server_encoder(
        data[: args.max_samples] if args.max_samples > 0 else data,
        args,
        device,
    )

    decoder_train_x, decoder_train_y = collect_observed_pairs(
        decoder_raw, server_encoder, args, device
    )
    eval_x, eval_y = collect_observed_pairs(eval_raw, server_encoder, args, device)
    member_x, member_y = collect_observed_pairs(target_raw, server_encoder, args, device)

    decoder_train_x, [eval_x, member_x] = standardize_messages(
        decoder_train_x, [eval_x, member_x]
    )

    decoder = ReconstructionDecoder(
        input_dim=decoder_train_x.size(1),
        hidden_dim=args.decoder_hidden_dim,
        output_dim=decoder_train_y.size(1),
    ).to(device)

    if args.decoder_checkpoint and os.path.exists(args.decoder_checkpoint):
        state = torch.load(args.decoder_checkpoint, map_location=device)
        decoder.load_state_dict(state["decoder"])
        print(f"Loaded decoder checkpoint from {args.decoder_checkpoint}")
    else:
        decoder_train_loader = build_tensor_loader(
            decoder_train_x,
            decoder_train_y,
            args.decoder_batch_size,
            shuffle=True,
        )
        decoder_eval_loader = build_tensor_loader(
            eval_x,
            eval_y,
            args.decoder_batch_size,
            shuffle=False,
        )
        train_decoder(decoder, decoder_train_loader, decoder_eval_loader, args, device)

        if args.decoder_checkpoint:
            torch.save({"decoder": decoder.state_dict()}, args.decoder_checkpoint)
            print(f"Saved decoder checkpoint to {args.decoder_checkpoint}")

    decoder_train_loader = build_tensor_loader(
        decoder_train_x, decoder_train_y, args.decoder_batch_size, shuffle=False
    )
    decoder_eval_loader = build_tensor_loader(
        eval_x, eval_y, args.decoder_batch_size, shuffle=False
    )
    member_loader = build_tensor_loader(
        member_x, member_y, args.decoder_batch_size, shuffle=False
    )

    train_metrics = evaluate_decoder(decoder, decoder_train_loader, device)
    eval_metrics = evaluate_decoder(decoder, decoder_eval_loader, device)
    member_metrics = evaluate_decoder(decoder, member_loader, device)

    results = {
        "attack_type": "private_code_reconstruction",
        "direction": args.direction,
        "privacy_scheme": args.privacy_scheme,
        "attacker_knows_encoder": False,
        "flip_epsilon": args.flip_epsilon,
        "num_hash_bits": args.num_hash_bits,
        "private_code_dim": int(decoder_train_x.size(1)),
        "target_raw_size": len(target_raw),
        "decoder_train_size": len(decoder_raw),
        "eval_raw_size": len(eval_raw),
        "decoder_train_metrics": train_metrics,
        "decoder_eval_metrics": eval_metrics,
        "target_member_metrics": member_metrics,
    }

    report_path = os.path.join(
        args.save_dir,
        f"recon_private_{args.direction}_{args.privacy_scheme.lower()}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Private-code reconstruction summary")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
