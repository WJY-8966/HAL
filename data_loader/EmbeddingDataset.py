
import torch
from torch.utils.data import Dataset
import random
import copy
from data_loader.utils import build_pseudo_aligned_dataset


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
                # Replace with incorrect text
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