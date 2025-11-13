import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from typing import List, Dict

# For text processing
from transformers import AutoTokenizer
import collections

# --- 1. 数据下载和准备 (使用 Hugging Face datasets) ---
# 指定你想要下载和缓存数据集的目录
# 请替换为你实际的路径
CUSTOM_CACHE_DIR = "/data/zhouxiaokai/data/dataset/Flicker30k"
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True) # 确保目录存在

print(f"正在尝试从 Hugging Face Hub 加载 Flickr30K 数据集到 '{CUSTOM_CACHE_DIR}'...")
try:
    hf_dataset = load_dataset("nlphuji/flickr30k", split='train', cache_dir=CUSTOM_CACHE_DIR)
    print("Flickr30K 数据集加载成功！")
    print(f"数据集大小: {len(hf_dataset)} 样本")
    print(f"一个样本示例: {hf_dataset[0]}")
except Exception as e:
    print(f"从 Hugging Face Hub 加载数据集失败: {e}")
    print("请确保你已安装 `datasets` 库: `pip install datasets`")
    print("如果网络问题，可能需要多尝试几次。")
    exit()

# Hugging Face 数据集的结构:
# 每一个样本 (dictionary) 包含 'image' (PIL.Image.Image) 和 'text' (list of 5 strings)

# --- 2. 文本处理：词汇表 (Vocabulary) 和 Tokenizer ---
print("\n正在加载文本分词器...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("分词器加载成功！")

# 辅助函数：将一个列表的句子分词并转换为ID
def tokenize_captions(captions: List[str], tokenizer):
    encoded_captions = []
    for caption in captions:
        encoded = tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False
        )['input_ids']
        encoded_captions.append(encoded)
    return encoded_captions

# --- 3. 创建自定义 PyTorch Dataset 类 ---
class Flickr30KMultiModalDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform=None):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample['image']
        captions = sample['text']

        if self.transform:
            image = self.transform(image)

        # 随机选择一个字幕
        selected_caption = captions[torch.randint(0, len(captions), (1,)).item()]
        tokenized_caption = self.tokenizer.encode_plus(
            selected_caption,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False
        )['input_ids']

        return image, torch.tensor(tokenized_caption, dtype=torch.long)

# --- 4. 图像预处理定义 ---
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 5. 自定义 collate_fn (用于处理变长序列) ---
def custom_collate_fn(batch: List[tuple]):
    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images, dim=0)

    max_len = max([len(cap) for cap in captions])

    padded_captions = torch.full((len(captions), max_len),
                                 fill_value=tokenizer.pad_token_id,
                                 dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap

    attention_mask = (padded_captions != tokenizer.pad_token_id).long()

    return images, padded_captions, attention_mask


# --- 6. 实例化 Dataset 和 DataLoader ---
if __name__ == '__main__':
    # 创建数据集实例
    print("\n正在创建多模态数据集实例...")
    multi_modal_dataset = Flickr30KMultiModalDataset(hf_dataset=hf_dataset,
                                                     tokenizer=tokenizer,
                                                     transform=image_transforms)
    print(f"成功创建多模态数据集，包含 {len(multi_modal_dataset)} 样本。")

    # 创建 DataLoader
    batch_size = 8
    multi_modal_dataloader = DataLoader(multi_modal_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=2,
                                        collate_fn=custom_collate_fn)

    # --- 7. 遍历 DataLoader 进行数据演示 ---
    print("\n--- 遍历 DataLoader 示例 ---")
    for i, (images, captions_padded, attention_mask) in enumerate(multi_modal_dataloader):
        print(f"批次 {i+1}:")
        print(f"  图像批次形状: {images.shape}")
        print(f"  字幕批次形状 (padding 后): {captions_padded.shape}")
        print(f"  注意力掩码形状: {attention_mask.shape}")

        print(f"  第一个图像的解码字幕: '{tokenizer.decode(captions_padded[0], skip_special_tokens=True)}'")
        print(f"  第一个图像的原始 token IDs: {captions_padded[0].tolist()}")
        print(f"  第一个图像的注意力掩码: {attention_mask[0].tolist()}")

        if i == 1:
            break

    print("\n数据加载和预处理完成。你可以使用这些批次数据来训练你的多模态模型。")