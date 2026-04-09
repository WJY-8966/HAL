import argparse
import json
import math
import os
from datetime import datetime
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.OrthogonalProjection import (
    estimate_shift,
    generate_orthogonal_lsh_projections,
    lsh_hash_bits,
    normalize_embeddings,
)
from trainer.PrivateHamming import bit_flip_matrix_torch
from trainer.TPOneHot import encode_tpoh_torch, generate_tpoh_hashes


DEFAULT_IMAGE_ROOT = "/data/wangjiayi/dataset/dataset/Flicker30k/Images"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an embedding-to-image decoder on an auxiliary split and reconstruct target images using only embeddings."
    )
    parser.add_argument(
        "--pretrained_file",
        type=str,
        default="/data/wangjiayi/HAL/pretrained_embeddings/Flicker30k/clip_embeddings.pt",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional target indices to attack. If omitted, attack all non-auxiliary samples.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help="Directory containing Flickr30k images for auxiliary training and final evaluation only.",
    )
    parser.add_argument("--aux_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["embedding", "tponehot_binary"],
        default="embedding",
        help="Choose whether to reconstruct from raw CLIP image embeddings or HAL TPOneHot+bitflipping binary codes.",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--num_hash_bits", type=int, default=512)
    parser.add_argument("--flip_epsilon", type=float, default=0.1)
    parser.add_argument("--direction", type=str, choices=["i2t", "t2i"], default="i2t")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pixel_l1_weight", type=float, default=1.0)
    parser.add_argument("--pixel_mse_weight", type=float, default=1.0)
    parser.add_argument("--embed_weight", type=float, default=0.5)
    parser.add_argument("--structure_l1_weight", type=float, default=1.0)
    parser.add_argument("--structure_mse_weight", type=float, default=0.5)
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="VGG 感知损失权重（0 表示禁用）。",
    )
    parser.add_argument("--decoder_checkpoint", type=str, default="")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/wangjiayi/HAL/attack_outputs",
        help="输出根目录；实际结果会写入子目录 reconstruct_<input_type>_<YYYYMMDD_HHMMSS>。",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_clip():
    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError("未找到 clip 包，请先在当前环境安装 openai-clip。") from exc
    return clip


def load_embedding_bank(pretrained_file):
    data = torch.load(pretrained_file, map_location="cpu")
    if not isinstance(data, list) or not data:
        raise ValueError("pretrained_file 必须是非空 list。")
    for key in ("image_embedding", "img_path"):
        if key not in data[0]:
            raise ValueError(f"样本中缺少字段 `{key}`")
    return data


def normalize_embedding(embedding):
    return F.normalize(embedding.float(), dim=0)


def resolve_img_path(raw_path, image_root):
    basename = os.path.basename(raw_path)
    candidate = os.path.join(image_root, basename)
    if os.path.exists(candidate):
        return candidate
    return ""


def load_image_to_tensor(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1)


def save_tensor_image(image_tensor, save_path):
    image = image_tensor.detach().cpu().clamp(0, 1)
    if image.dim() == 4:
        image = image[0]
    image_np = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(image_np).save(save_path)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class AuxiliaryImageDataset(Dataset):
    def __init__(self, samples, image_root, image_size, input_key):
        self.items = []
        self.input_key = input_key
        for item in samples:
            img_path = resolve_img_path(item.get("img_path", ""), image_root)
            if not img_path:
                continue
            self.items.append(
                {
                    "input_feature": item[input_key].float(),
                    "img_path": img_path,
                }
            )
        self.image_size = image_size
        if not self.items:
            raise ValueError("辅助数据集为空，请检查 image_root 路径和数据文件。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = load_image_to_tensor(item["img_path"], self.image_size)
        return {
            "input_feature": item["input_feature"],
            "image": image,
            "img_path": item["img_path"],
        }


def make_group_norm(num_channels):
    for num_groups in (32, 16, 8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConditionalResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = make_group_norm(channels)
        self.norm2 = make_group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x, cond):
        gamma, beta = self.cond_proj(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        residual = x
        x = self.norm1(x)
        x = x * (1.0 + gamma) + beta
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class FrozenResNetImagePrior(nn.Module):
    """Frozen ImageNet ResNet18 encoder used as a natural-image prior."""

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        backbone = torchvision_models.resnet18(
            weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for param in self.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image):
        x = (image - self.mean) / self.std
        stem = self.stem(x)  # 112 x 112
        x = self.maxpool(stem)
        layer1 = self.layer1(x)  # 56 x 56
        layer2 = self.layer2(layer1)  # 28 x 28
        layer3 = self.layer3(layer2)  # 14 x 14
        layer4 = self.layer4(layer3)  # 7 x 7
        return {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }


class RefineUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.skip_proj = (
            nn.Conv2d(skip_channels, out_channels, kernel_size=1)
            if skip_channels > 0
            else None
        )
        self.block = ConditionalResBlock(out_channels, cond_dim)

    def forward(self, x, cond, skip=None):
        x = self.up(x)
        if skip is not None and self.skip_proj is not None:
            x = x + self.skip_proj(skip)
        x = self.block(x, cond)
        return x


class FeatureToImageDecoder(nn.Module):
    def __init__(self, input_dim, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.structure_size = 56

        self.structure_fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512 * 7 * 7),
            nn.GELU(),
        )
        self.structure_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            make_group_norm(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            make_group_norm(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            make_group_norm(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            make_group_norm(64),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.prior_encoder = FrozenResNetImagePrior()
        self.bottleneck = ConditionalResBlock(512, input_dim)
        self.up1 = RefineUpBlock(512, 256, 256, input_dim)
        self.up2 = RefineUpBlock(256, 128, 128, input_dim)
        self.up3 = RefineUpBlock(128, 64, 64, input_dim)
        self.up4 = RefineUpBlock(64, 64, 64, input_dim)
        self.up5 = RefineUpBlock(64, 0, 32, input_dim)
        self.final_block = ConditionalResBlock(32, input_dim)
        self.final_head = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        x = self.structure_fc(feature)
        x = x.view(feature.size(0), 512, 7, 7)
        structure_image = self.structure_decoder(x)

        coarse_image = F.interpolate(
            structure_image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        prior_feats = self.prior_encoder(coarse_image)
        x = self.bottleneck(prior_feats["layer4"], feature)
        x = self.up1(x, feature, prior_feats["layer3"])
        x = self.up2(x, feature, prior_feats["layer2"])
        x = self.up3(x, feature, prior_feats["layer1"])
        x = self.up4(x, feature, prior_feats["stem"])
        x = self.up5(x, feature)
        x = self.final_block(x, feature)
        detail_delta = torch.tanh(self.final_head(x))
        final_image = torch.clamp(coarse_image + 0.25 * detail_delta, 0.0, 1.0)
        return {
            "structure_image": structure_image,
            "final_image": final_image,
        }


class FrozenClipImageEncoder(nn.Module):
    def __init__(self, clip_model_name, device):
        super().__init__()
        clip = load_clip()
        model, _ = clip.load(clip_model_name, device=device, jit=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def forward(self, image):
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - self.clip_mean) / self.clip_std
        embedding = self.model.encode_image(image)
        return F.normalize(embedding.float(), dim=1)


class VGGPerceptualLoss(nn.Module):
    """用 VGG16 前 16 层（relu3_3）的特征 L1 距离作为感知损失。
    输入图像应为 [0, 1] 范围的 float 张量，形状 (N, 3, H, W)。
    """

    # ImageNet 归一化参数
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, device):
        super().__init__()
        vgg = torchvision_models.vgg16(
            weights=torchvision_models.VGG16_Weights.IMAGENET1K_V1
        )
        # 取 relu3_3 之前的所有层（含 relu3_3，共 16 层）
        self.features = vgg.features[:16].to(device).eval()
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor(self._MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._STD, device=device).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.features(pred_n), self.features(target_n))


def build_auxiliary_split(data, target_indices, aux_ratio, seed):
    target_set = set(target_indices)
    candidates = [idx for idx in range(len(data)) if idx not in target_set]
    aux_count = max(1, int(len(candidates) * aux_ratio))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    aux_indices = sorted(candidates[:aux_count])
    return [data[idx] for idx in aux_indices], aux_indices


def build_attack_indices(num_samples, auxiliary_indices, requested_indices):
    auxiliary_set = set(auxiliary_indices)
    if requested_indices is None:
        return [idx for idx in range(num_samples) if idx not in auxiliary_set]

    invalid = [idx for idx in requested_indices if idx < 0 or idx >= num_samples]
    if invalid:
        raise IndexError(f"indices 越界: {invalid}，合法范围是 [0, {num_samples - 1}]")

    overlap = [idx for idx in requested_indices if idx in auxiliary_set]
    if overlap:
        raise ValueError(f"这些 indices 落在辅助集内，不能同时作为攻击目标: {overlap}")

    return sorted(set(requested_indices))


def stack_modalities(data):
    img = torch.stack([item["image_embedding"] for item in data]).float()
    txt = torch.stack([item["text_embedding"] for item in data]).float()
    return img, txt


def build_hidden_server_encoder(all_data, args, device):
    img_embeddings, txt_embeddings = stack_modalities(all_data)
    img_embeddings = normalize_embeddings(img_embeddings.to(device))
    txt_embeddings = normalize_embeddings(txt_embeddings.to(device))

    shift = estimate_shift(img_embeddings, txt_embeddings)
    projections = generate_orthogonal_lsh_projections(
        shift,
        dim=img_embeddings.size(1),
        num_vecs=args.num_hash_bits,
    )
    h0, h1, code_dim = generate_tpoh_hashes(n=args.num_hash_bits, seed=args.seed)
    return {
        "projections": projections,
        "h0": h0,
        "h1": h1,
        "code_dim": code_dim,
    }


def server_encode_messages(embeddings, server_encoder, args, device):
    embeddings = normalize_embeddings(embeddings.to(device))
    bits = lsh_hash_bits(embeddings, server_encoder["projections"])
    encoded = encode_tpoh_torch(
        bits,
        server_encoder["h0"],
        server_encoder["h1"],
        server_encoder["code_dim"],
    )
    flipped = bit_flip_matrix_torch(encoded, args.flip_epsilon, seed=args.seed)
    return flipped.float()


def attach_attack_codes(data, server_encoder, args, device):
    img_embeddings, txt_embeddings = stack_modalities(data)
    source_embeddings = img_embeddings if args.direction == "i2t" else txt_embeddings
    attack_codes = server_encode_messages(source_embeddings, server_encoder, args, device).cpu()

    enriched = []
    for item, code in zip(data, attack_codes):
        new_item = dict(item)
        new_item["attack_code"] = code
        enriched.append(new_item)
    return enriched


def attach_raw_embeddings(data):
    enriched = []
    for item in data:
        new_item = dict(item)
        new_item["raw_embedding"] = normalize_embedding(item["image_embedding"]).float()
        enriched.append(new_item)
    return enriched


def split_train_val(dataset, val_ratio, seed):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("辅助数据太少，无法切分出训练集与验证集。")
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_losses(pred_output, target_image, target_embedding, clip_encoder, args, perceptual_loss_fn=None):
    pred_image = pred_output["final_image"]
    structure_image = pred_output["structure_image"]
    target_structure = F.interpolate(
        target_image,
        size=structure_image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    loss_structure_l1 = F.l1_loss(structure_image, target_structure)
    loss_structure_mse = F.mse_loss(structure_image, target_structure)
    loss_l1 = F.l1_loss(pred_image, target_image)
    loss_mse = F.mse_loss(pred_image, target_image)
    pred_embedding = clip_encoder(pred_image)
    loss_embed = 1.0 - F.cosine_similarity(pred_embedding, target_embedding, dim=1).mean()
    loss = (
        args.structure_l1_weight * loss_structure_l1
        + args.structure_mse_weight * loss_structure_mse
        + args.pixel_l1_weight * loss_l1
        + args.pixel_mse_weight * loss_mse
        + args.embed_weight * loss_embed
    )
    loss_perceptual = torch.zeros(1, device=pred_image.device)
    if perceptual_loss_fn is not None and args.perceptual_weight > 0:
        loss_perceptual = perceptual_loss_fn(pred_image, target_image)
        loss = loss + args.perceptual_weight * loss_perceptual
    return (
        loss,
        loss_l1,
        loss_mse,
        loss_embed,
        loss_perceptual,
        loss_structure_l1,
        loss_structure_mse,
    )


def run_epoch(model, dataloader, optimizer, clip_encoder, args, device, train, perceptual_loss_fn=None):
    if train:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "embed": 0.0,
        "perceptual": 0.0,
        "structure_l1": 0.0,
        "structure_mse": 0.0,
        "count": 0,
    }
    phase = "train" if train else "val"
    progress = tqdm(dataloader, desc=f"{phase}", leave=False)
    for batch in progress:
        input_features = batch["input_feature"].to(device)
        images = batch["image"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(input_features)
            if args.input_type == "embedding":
                target_embedding = input_features
            else:
                target_embedding = clip_encoder(images)
            (
                loss,
                loss_l1,
                loss_mse,
                loss_embed,
                loss_perceptual,
                loss_structure_l1,
                loss_structure_mse,
            ) = compute_losses(
                pred,
                images,
                target_embedding,
                clip_encoder,
                args,
                perceptual_loss_fn=perceptual_loss_fn,
            )
            if train:
                loss.backward()
                optimizer.step()

        bs = input_features.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["l1"] += float(loss_l1.item()) * bs
        totals["mse"] += float(loss_mse.item()) * bs
        totals["embed"] += float(loss_embed.item()) * bs
        totals["perceptual"] += float(loss_perceptual.item()) * bs
        totals["structure_l1"] += float(loss_structure_l1.item()) * bs
        totals["structure_mse"] += float(loss_structure_mse.item()) * bs
        totals["count"] += bs

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            l1=f"{loss_l1.item():.4f}",
            embed=f"{loss_embed.item():.4f}",
            perc=f"{loss_perceptual.item():.4f}",
            s_l1=f"{loss_structure_l1.item():.4f}",
        )

    count = max(1, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "l1": totals["l1"] / count,
        "mse": totals["mse"] / count,
        "embed": totals["embed"] / count,
        "perceptual": totals["perceptual"] / count,
        "structure_l1": totals["structure_l1"] / count,
        "structure_mse": totals["structure_mse"] / count,
    }


def train_decoder(model, train_loader, val_loader, clip_encoder, args, device, perceptual_loss_fn=None):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, clip_encoder, args, device,
            train=True, perceptual_loss_fn=perceptual_loss_fn,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, clip_encoder, args, device,
            train=False, perceptual_loss_fn=perceptual_loss_fn,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"[decoder] epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_l1={val_metrics['l1']:.6f} "
            f"val_structure_l1={val_metrics['structure_l1']:.6f} "
            f"val_embed={val_metrics['embed']:.6f} "
            f"val_perc={val_metrics['perceptual']:.6f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("decoder 训练失败，没有得到有效参数。")
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def infer_image(model, input_feature, device):
    model.eval()
    input_feature = input_feature.float().unsqueeze(0).to(device)
    pred = model(input_feature)
    return pred["final_image"].cpu(), pred["structure_image"].cpu()


def psnr_from_mse(mse_value):
    mse_value = max(mse_value, 1e-12)
    return 10.0 * math.log10(1.0 / mse_value)


def ssim_from_images(pred, gt):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean().item()
    mu_y = gt.mean().item()
    sigma_x = pred.var(unbiased=False).item()
    sigma_y = gt.var(unbiased=False).item()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean().item()
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return numerator / max(denominator, 1e-12)


@torch.no_grad()
def evaluate_reconstruction(pred_image, target_item, clip_encoder, args, sample_dir):
    target_path = resolve_img_path(target_item.get("img_path", ""), args.image_root)
    metrics = {
        "ground_truth_path": target_path,
        "mse": None,
        "l1": None,
        "mae": None,
        "psnr": None,
        "ssim": None,
        "embedding_cosine_to_target": None,
    }
    if not target_path:
        return metrics

    gt = load_image_to_tensor(target_path, args.image_size).unsqueeze(0)
    pred = pred_image.clamp(0, 1)

    mse = F.mse_loss(pred, gt).item()
    l1 = F.l1_loss(pred, gt).item()
    pred_embed = clip_encoder(pred.to(next(clip_encoder.parameters()).device)).cpu()
    target_embed = normalize_embedding(target_item["image_embedding"]).unsqueeze(0)
    emb_cos = F.cosine_similarity(pred_embed, target_embed, dim=1).mean().item()

    gt_copy = os.path.join(sample_dir, "ground_truth.jpg")
    shutil.copy2(target_path, gt_copy)

    metrics.update(
        {
            "ground_truth_path": target_path,
            "ground_truth_copy": gt_copy,
            "mse": mse,
            "l1": l1,
            "mae": l1,
            "psnr": psnr_from_mse(mse),
            "ssim": ssim_from_images(pred, gt),
            "embedding_cosine_to_target": emb_cos,
        }
    )
    return metrics


def summarize_attack_results(results):
    metric_names = ["mse", "l1", "mae", "psnr", "ssim", "embedding_cosine_to_target"]
    summary = {"num_attacked": len(results)}
    for name in metric_names:
        values = [
            item["evaluation"][name]
            for item in results
            if item.get("evaluation", {}).get(name) is not None
        ]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values))
        else:
            summary[f"{name}_mean"] = None
            summary[f"{name}_std"] = None
    return summary


def print_attack_summary(summary):
    print("=" * 60)
    print("Attack metric summary")
    print(f"num_attacked: {summary['num_attacked']}")
    metric_names = ["mse", "l1", "mae", "psnr", "ssim", "embedding_cosine_to_target"]
    for name in metric_names:
        mean_key = f"{name}_mean"
        std_key = f"{name}_std"
        print(f"{mean_key}: {summary.get(mean_key)}")
        print(f"{std_key}: {summary.get(std_key)}")


def main():
    args = parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(
        args.save_dir, f"reconstruct_{args.input_type}_{run_stamp}"
    )
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    if not os.path.exists(args.image_root):
        raise ValueError(f"image_root 不存在: {args.image_root}")

    raw_data = load_embedding_bank(args.pretrained_file)
    if args.input_type == "tponehot_binary":
        server_encoder = build_hidden_server_encoder(raw_data, args, device)
        data = attach_attack_codes(raw_data, server_encoder, args, device)
        input_key = "attack_code"
    else:
        data = attach_raw_embeddings(raw_data)
        input_key = "raw_embedding"

    requested_indices = args.indices
    excluded_from_aux = requested_indices if requested_indices is not None else []
    auxiliary_samples, auxiliary_indices = build_auxiliary_split(
        data, excluded_from_aux, args.aux_ratio, args.seed
    )
    attack_indices = build_attack_indices(len(data), auxiliary_indices, requested_indices)
    auxiliary_dataset = AuxiliaryImageDataset(
        auxiliary_samples,
        image_root=args.image_root,
        image_size=args.image_size,
        input_key=input_key,
    )
    train_dataset, val_dataset = split_train_val(auxiliary_dataset, args.val_ratio, args.seed)
    train_loader = create_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    input_dim = int(data[0][input_key].numel())
    decoder = FeatureToImageDecoder(input_dim=input_dim).to(device)
    clip_encoder = FrozenClipImageEncoder(args.clip_model, device).to(device)

    perceptual_loss_fn = None
    if args.perceptual_weight > 0:
        perceptual_loss_fn = VGGPerceptualLoss(device).to(device)
        print(f"VGG 感知损失已启用，权重={args.perceptual_weight}")

    if args.decoder_checkpoint and os.path.exists(args.decoder_checkpoint):
        state = torch.load(args.decoder_checkpoint, map_location=device)
        decoder.load_state_dict(state["decoder"])
        train_history = state.get("history", [])
        print(f"Loaded decoder checkpoint from {args.decoder_checkpoint}")
    else:
        train_history = train_decoder(
            decoder, train_loader, val_loader, clip_encoder, args, device,
            perceptual_loss_fn=perceptual_loss_fn,
        )
        if args.decoder_checkpoint:
            torch.save(
                {"decoder": decoder.state_dict(), "history": train_history},
                args.decoder_checkpoint,
            )
            print(f"Saved decoder checkpoint to {args.decoder_checkpoint}")

    run_summary = {
        "pretrained_file": args.pretrained_file,
        "image_root": args.image_root,
        "aux_ratio": args.aux_ratio,
        "input_type": args.input_type,
        "auxiliary_size": len(auxiliary_dataset),
        "auxiliary_indices": auxiliary_indices,
        "target_indices": attack_indices,
        "train_history": train_history,
        "results": [],
    }

    attack_progress = tqdm(attack_indices, desc="attack", leave=True)
    for target_index in attack_progress:
        target_item = data[target_index]
        sample_dir = os.path.join(args.save_dir, f"sample_{target_index:05d}")
        os.makedirs(sample_dir, exist_ok=True)

        pred_image, structure_image = infer_image(decoder, target_item[input_key], device)
        recon_path = os.path.join(sample_dir, "reconstructed.png")
        save_tensor_image(pred_image, recon_path)
        structure_path = os.path.join(sample_dir, "structure.png")
        save_tensor_image(
            F.interpolate(
                structure_image,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            ),
            structure_path,
        )

        evaluation = evaluate_reconstruction(
            pred_image,
            target_item,
            clip_encoder,
            args,
            sample_dir,
        )

        sample_summary = {
            "index": target_index,
            "caption": target_item.get("caption", ""),
            "stored_img_path": target_item.get("img_path", ""),
            "input_type": args.input_type,
            "input_dim": int(target_item[input_key].numel()),
            "reconstructed_path": recon_path,
            "structure_path": structure_path,
            "evaluation": evaluation,
        }
        save_json(sample_summary, os.path.join(sample_dir, "summary.json"))
        run_summary["results"].append(sample_summary)
        metric = sample_summary["evaluation"].get("embedding_cosine_to_target")
        attack_progress.set_postfix(
            index=target_index,
            emb_cos=f"{metric:.4f}" if metric is not None else "NA",
        )

    run_summary["aggregate_metrics"] = summarize_attack_results(run_summary["results"])
    summary_path = os.path.join(args.save_dir, "run_summary.json")
    save_json(run_summary, summary_path)
    print_attack_summary(run_summary["aggregate_metrics"])
    print(f"Saved attack outputs to {args.save_dir}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
