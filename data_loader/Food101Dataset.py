"""
Food101 Dataset Loader for multimodal learning
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import random

class Food101Dataset(Dataset):
    def __init__(self,
                 root='/data/zhouxiaokai/data/dataset/Food-101/',
                 split='train',
                 transform=None,
                 tokenizer=None,
                 max_length=64,
                 template_type='simple'):
        """
        Args:
            root (string): Root directory of the Food-101 dataset
            split (string): 'train' or 'test'
            transform: Optional transform for images
            tokenizer: Optional tokenizer for text
            max_length: Maximum length for tokenized text
            template_type (string): Type of text template to use
                - 'simple': "a photo of {food_name}"
                - 'detailed': "a photo of {food_name}, a type of food"
                - 'question': "what type of food is shown in this image? {food_name}"
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_type = template_type

        # Load Food-101 dataset
        self.dataset = datasets.Food101(
            root=root,
            split=split,
            download=False  # Assuming already downloaded
        )

        # Create class name mapping (index to name)
        self.idx_to_class = {
            i: name.replace('_', ' ') for i, name in enumerate(self.dataset.classes)
        }

        # Create text templates
        self.templates = {
            'simple': "a photo of {}",
            'detailed': "a photo of {}, a type of food",
            'question': "what type of food is shown in this image? {}"
        }

    def _get_text_description(self, class_name):
        """Generate text description based on template type"""
        template = self.templates.get(self.template_type, self.templates['simple'])
        return template.format(class_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image, label = self.dataset[idx]
        
        # Get class name and create text description
        class_name = self.idx_to_class[label]
        text = self._get_text_description(class_name)

        # Apply image transforms if any
        if self.transform:
            image = self.transform(image)

        # Create output dictionary
        output = {
            'image': image,
            'text': text,
            'label': label,
            'class_name': class_name
        }

        # Apply tokenization if tokenizer is provided
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            output['input_ids'] = tokens['input_ids'].squeeze(0)
            output['attention_mask'] = tokens['attention_mask'].squeeze(0)

        return output

def food101_collate_fn(batch):
    """
    Custom collate function for Food101 dataset
    """
    images = []
    texts = []
    labels = []
    class_names = []
    input_ids = []
    attention_masks = []
    has_tokens = 'input_ids' in batch[0]

    for item in batch:
        images.append(item['image'])
        texts.append(item['text'])
        labels.append(item['label'])
        class_names.append(item['class_name'])
        
        if has_tokens:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])

    # Stack all gathered data
    batch_out = {
        'image': torch.stack(images) if torch.is_tensor(images[0]) else images,
        'text': texts,
        'label': torch.tensor(labels),
        'class_name': class_names
    }

    if has_tokens:
        batch_out['input_ids'] = torch.stack(input_ids)
        batch_out['attention_mask'] = torch.stack(attention_masks)

    return batch_out

if __name__ == "__main__":
    from torchvision import transforms
    from transformers import AutoTokenizer
    import torch

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset
    dataset = Food101Dataset(
        root='/data/zhouxiaokai/data/dataset/Food-101/',
        split='train',
        transform=transform,
        tokenizer=tokenizer,
        template_type='detailed'
    )

    # Test the dataset
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print("\nSample data:")
    print(f"Class name: {sample['class_name']}")
    print(f"Text description: {sample['text']}")
    print(f"Label: {sample['label']}")
    print(f"Image shape: {sample['image'].shape}")
    if 'input_ids' in sample:
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")

    # Test with DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=food101_collate_fn
    )
    
    # Get a batch
    batch = next(iter(loader))
    print("\nBatch data:")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch text samples: {batch['text']}")
    print(f"Batch labels shape: {batch['label'].shape}")
    if 'input_ids' in batch:
        print(f"Batch input IDs shape: {batch['input_ids'].shape}")
        print(f"Batch attention mask shape: {batch['attention_mask'].shape}") 