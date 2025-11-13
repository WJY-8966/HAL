"""
MSCOCO Caption Dataset Loader
"""
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import copy
from data_loader.utils import build_pseudo_aligned_dataset

class CocoClipDataset(Dataset):
    def __init__(self, img_dir, caption_ann_file, category_ann_file):
        """
        img_dir: 'coco/train2017'
        caption_ann_file: captions_train2017.json
        category_ann_file: instances_train2017.json
        """
        self.img_dir = img_dir

        # load caption
        with open(caption_ann_file, 'r') as f:
            caption_data = json.load(f)
        id_to_filename = {img['id']: img['file_name'] for img in caption_data['images']}
        image_id_to_caption = {}
        for ann in caption_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image_id_to_caption.setdefault(image_id, []).append(caption)

        # load category
        with open(category_ann_file, 'r') as f:
            category_data = json.load(f)
        image_id_to_category = {}
        for ann in category_data['annotations']:
            image_id_to_category[ann['image_id']] = ann['category_id']

        # get all category_id and encoding
        all_category_ids = list(set(image_id_to_category.values()))
        category_id_to_label = {cid: idx for idx, cid in enumerate(sorted(all_category_ids))}

        # construct samples
        self.samples = []
        for image_id, file_name in id_to_filename.items():
            if image_id in image_id_to_caption and image_id in image_id_to_category:
                caption = image_id_to_caption[image_id][0]
                category = category_id_to_label[image_id_to_category[image_id]]  # 使用映射后的 label
                img_path = os.path.join(img_dir, file_name)
                self.samples.append((img_path, caption, category))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, category = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return {
            'image': image,
            'caption': caption,
            'img_path': img_path,
            'category': category
        }

class CocoCaptionDataset(Dataset):
    def __init__(self,
                 img_dir,
                 ann_file,
                 transform=None,
                 tokenizer=None,
                 max_length=64):
        """
        - img_dir: image path 'coco/train2017'
        - ann_file: JSON path， 'coco/annotations/captions_train2017.json'
        - transform: image transform
        - tokenizer:text tokenizer（transformers-AutoTokenizer）
        - max_length: maximum length of tokenized captions
        """
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load json
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # construct image_id -> file_name mapping
        self.image_id_to_file = {
            img['id']: img['file_name'] for img in data['images']
        }

        # construct (image_path, caption)
        self.samples = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            file_name = self.image_id_to_file[image_id]
            img_path = os.path.join(self.img_dir, file_name)
            self.samples.append((img_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        # load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # process text
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            return {
                'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'caption': caption,
                'img_path': img_path
            }
        else:
            return {
                'image': image,
                'caption': caption,
                'img_path': img_path
            }

class CocoAlignedDataset(CocoCaptionDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['label'] = 1  # 对齐
        return sample

class CocoUnalignedDataset(CocoCaptionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # save caption
        self.all_captions = [cap for _, cap in self.samples]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # randomly select a different caption
        original_caption = sample['caption']
        while True:
            wrong_caption = random.choice(self.all_captions)
            if wrong_caption != original_caption:
                break

        if self.tokenizer:
            tokens = self.tokenizer(
                wrong_caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            sample['input_ids'] = input_ids
            sample['attention_mask'] = attention_mask
            sample['caption'] = wrong_caption
        else:
            sample['caption'] = wrong_caption

        sample['label'] = 0
        return sample


class EmbeddingDataset(Dataset):
    def __init__(self, data, add_negative= True, negative_ratio=0.1, mode='aligned', seed = 2025):
        """
        mode: 'aligned' or 'unaligned'
        """
        # self.data = data
        self.mode = mode
        self.rng = random.Random(seed)
        self.add_negative = add_negative
        self.negative_ratio = negative_ratio

        self.data = data

        if self.mode == 'balanced':
            self.data = self._build_balanced_dataset(self.data)
        elif self.mode == 'unaligned':
            self.data = self._build_unaligned_dataset(self.data)
        elif self.mode == 'unaligned_IEMOCAP':
            self.data = self._build_unaligned_IEMOCAP(self.data)
        elif self.mode == 'balanced_unaligned':
            self.data = self._build_balanced_unaligned_dataset(self.data)
        elif self.mode == 'balanced_label_unaligned':
            self.data = self._build_balanced_label_unaligned_dataset(self.data)
        elif self.mode == 'pseudo_aligned':
            self.data = self._build_pseudo_aligned_dataset(self.data)

    def _build_balanced_dataset(self, data):
        total_n = len(data)
        num_neg = int(total_n * self.negative_ratio)
        num_pos = total_n - num_neg

        indices = list(range(total_n))
        self.rng.shuffle(indices)

        neg_indices = set(indices[:num_neg])
        text_pool = [d['text_embedding'] for d in data]

        balanced_data = []

        for i in range(total_n):
            item = data[i]
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']
            category = item['category']

            if i in neg_indices:
                # Replace it with incorrect text
                while True:
                    rand_text = self.rng.choice(text_pool)
                    if not torch.equal(rand_text, text_embedding):
                        break
                text_embedding = rand_text
                label = 0
            else:
                label = 1

            balanced_data.append({
                'image_embedding': image_embedding,
                'text_embedding': text_embedding,
                'category': category,
                'is_aligned': label
            })

        return balanced_data

    def _build_balanced_unaligned_dataset(self, data):
        total_n = len(data)
        num_neg = int(total_n * self.negative_ratio)
        num_pos = total_n - num_neg

        indices = list(range(total_n))
        self.rng.shuffle(indices)
        text_pool = [d['text_embedding'] for d in data]

        balanced_data = []

        # Construct "fake positives" (mismatched but labeled as 1)
        for i in indices[:num_pos]:
            item = data[i]
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']  # keep mismatched but pretend matched
            balanced_data.append({
                'image_embedding': image_embedding,
                'text_embedding': text_embedding,
                'category': item['category'],
                'is_aligned': 1
            })

        # Construct true negatives (mismatched and labeled 0)
        for i in indices[num_pos:num_pos + num_neg]:
            item = data[i]
            image_embedding = item['image_embedding']
            while True:
                rand_text = self.rng.choice(text_pool)
                if not torch.equal(rand_text, item['text_embedding']):
                    break
            balanced_data.append({
                'image_embedding': image_embedding,
                'text_embedding': rand_text,
                'category': item['category'],
                'is_aligned': 0
            })

        return balanced_data

    def _build_pseudo_aligned_dataset(self, data):
        unaligned_data = self._build_unaligned_dataset(data)
        pseudo_aligned_data = build_pseudo_aligned_dataset(unaligned_data)

        return pseudo_aligned_data

    def _build_balanced_label_unaligned_dataset(self, data):
        balanced_label_data = self._build_balanced_dataset(data)

        unaligned_data = copy.deepcopy(balanced_label_data)
        text_embeddings = [item['text_embedding'] for item in unaligned_data]
        self.rng.shuffle(text_embeddings)
        for i, item in enumerate(unaligned_data):
            item['text_embedding'] = text_embeddings[i]

        return unaligned_data

    def _build_unaligned_dataset(self, data):

        unaligned_data = copy.deepcopy(data)
        # text_embeddings = [item['text_embedding'] for item in unaligned_data]
        # self.rng.shuffle(text_embeddings)
        text_embeddings = [item['text_embedding'] for item in unaligned_data]
        self.rng.shuffle(text_embeddings)
        for i, item in enumerate(unaligned_data):
            item['text_embedding'] = text_embeddings[i]

        return unaligned_data

    def _build_unaligned_IEMOCAP(self, data):
        unaligned_data = copy.deepcopy(data)
        text_embeddings = [item['text_embedding'] for item in unaligned_data]
        audio_embeddings = [item['audio_embedding'] for item in unaligned_data]
        self.rng.shuffle(text_embeddings)
        self.rng.shuffle(audio_embeddings)
        for i, item in enumerate(unaligned_data):
            item['text_embedding'] = text_embeddings[i]
            item['audio_embedding'] = audio_embeddings[i]
        return unaligned_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == 'aligned':
            label = 1
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']
            category = item['category']
        elif self.mode == 'unaligned':
            label = 0
            # image_embedding = item['image_embedding']
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']
            category = item['category']
        elif self.mode == 'aligned_IEMOCAP':
            label = 1
            video_embedding = item['video_embedding']
            text_embedding = item['text_embedding']
            audio_embedding = item['audio_embedding']
            category = item['category']
            return {'video_embedding': video_embedding.float(),'text_embedding': text_embedding.float(),
                    'audio_embedding': audio_embedding.float(),
                    'category': torch.tensor(category, dtype=torch.long),
                    'is_aligned': torch.tensor(label, dtype=torch.long)}
        elif self.mode == 'unaligned_IEMOCAP':
            label = 0
            video_embedding = item['video_embedding']
            text_embedding = item['text_embedding']
            audio_embedding = item['audio_embedding']
            category = item['category']
            return {'video_embedding': video_embedding.float(),'text_embedding': text_embedding.float(),
                    'audio_embedding': audio_embedding.float(),
                    'category': torch.tensor(category, dtype=torch.long),
                    'is_aligned': torch.tensor(label, dtype=torch.long)}
        elif self.mode == 'pseudo_aligned':
            label = 1
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']
            category = item['category']

            # category = item['category']
        elif self.mode in ['balanced', 'balanced_unaligned', 'balanced_label_unaligned']:
            image_embedding = item['image_embedding']
            text_embedding = item['text_embedding']
            category = item['category']
            label = item['is_aligned']

        return {
            'image_embedding': image_embedding.float(),
            'text_embedding': text_embedding.float(),
            'category': torch.tensor(category, dtype=torch.long),
            'is_aligned': torch.tensor(label, dtype=torch.long)
        }

def coco_collate_fn(batch):
    images = []
    input_ids = []
    attention_masks = []
    captions = []
    img_paths = []
    labels = []

    for sample in batch:
        # image
        images.append(sample['image'])

        # text
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])

        # others
        captions.append(sample['caption'])
        img_paths.append(sample['img_path'])
        labels.append(sample['label'])

    # Padding input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    images = torch.stack(images)

    return {
        'images': images,                          # [B, 3, H, W]
        'input_ids': input_ids,                    # [B, L]
        'attention_mask': attention_masks,         # [B, L]
        'captions': captions,                      # List[str]
        'img_paths': img_paths,                    # List[str]
        'labels': torch.tensor(labels),            # [B]
    }


def collate_fn(batch):
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    img_paths = [item['img_path'] for item in batch]

    images = torch.stack(images, dim=0)

    if 'input_ids' in batch[0]:
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': captions,
            'img_path': img_paths
        }
    else:
        return {
            'image': images,
            'caption': captions,
            'img_path': img_paths
        }

def clip_collate_fn(batch):
    # print(batch)
    images = [item['image'] for item in batch]  # 保持为 PIL list
    captions = [item['caption'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    categories = [item['category'] for item in batch]


    return {
        'image': images,
        'caption': captions,
        'img_path': img_paths,
        'category': categories
    }

def embedding_collate_fn(batch):
    img_embeds = torch.stack([x['image_embedding'] for x in batch])
    txt_embeds = torch.stack([x['text_embedding'] for x in batch])
    categories = torch.stack([x['category'] for x in batch])
    is_aligned = torch.stack([x['is_aligned'] for x in batch])

    return {
        'image_embedding': img_embeds,
        'text_embedding': txt_embeds,
        'category': categories ,
        'is_aligned': is_aligned,
    }
if __name__ == "__main__":
    # Example usage
    from torchvision import transforms
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    img_dir = '/MSCOCO/train2017'
    ann_file = '/MSCOCO/annotations/captions_train2017.json'

    # process image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Dataset
    dataset = CocoCaptionDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=image_transform,
        tokenizer=tokenizer,
        max_length=64
    )

    #  DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

