"""
Flickr30k Caption Dataset Loader with support for multiple captions per image
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import copy
from tqdm import tqdm

class Flickr30kDataset(Dataset):
    def __init__(self,
                 img_dir,
                 caption_file,
                 transform=None,
                 tokenizer=None,
                 max_length=64,
                 caption_strategy='first',
                 validate_images=False,
                 min_caption_length=4):
        """
        Args:
            img_dir (string): Directory with all the images
            caption_file (string): Path to the caption file
            transform: Optional transform to be applied on images
            tokenizer: Optional text tokenizer
            max_length: Maximum length of tokenized captions
            caption_strategy (string): Strategy for handling multiple captions
            validate_images (bool): Whether to validate all images
            min_caption_length (int): Minimum number of words in a valid caption
        """
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_strategy = caption_strategy
        self.min_caption_length = min_caption_length

        # Print initial directory check
        print(f"Checking image directory: {img_dir}")
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory does not exist: {img_dir}")

        # Read caption file with header and handle NaN values
        print(f"Reading caption file: {caption_file}")
        self.data = pd.read_csv(caption_file)
        
        # Remove rows with NaN values
        original_len = len(self.data)
        self.data = self.data.dropna(subset=['image', 'caption'])
        nan_removed = original_len - len(self.data)
        if nan_removed > 0:
            print(f"Removed {nan_removed} rows with NaN values")

        # Group captions by image
        self.image_to_captions = {}
        print("\nProcessing captions and images...")
        invalid_images = []
        missing_images = []
        invalid_captions = []

        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            try:
                # Extract image name and ensure it has the correct extension
                img_name = str(row['image']).strip()
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_name = f"{img_name}.jpg"

                caption = str(row['caption']).strip()

                # Skip empty strings
                if not img_name or not caption:
                    continue

                # Verify image exists
                img_path = os.path.join(self.img_dir, img_name)
                if not os.path.exists(img_path):
                    missing_images.append(img_name)
                    continue

                if img_name not in self.image_to_captions:
                    self.image_to_captions[img_name] = []
                self.image_to_captions[img_name].append(caption)
            except Exception as e:
                print(f"\nError processing row: {row}")
                print(f"Error details: {e}")
                invalid_images.append(img_name)
                continue

        # Create samples list based on caption strategy
        self.samples = self._create_samples()
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total images in caption file: {self.data['image'].nunique()}")
        print(f"Successfully loaded images: {len(self.image_to_captions)}")
        print(f"Total samples created: {len(self.samples)}")
        print(f"Average captions per image: {sum(len(caps) for caps in self.image_to_captions.values()) / len(self.image_to_captions) if self.image_to_captions else 0:.2f}")
        
        if missing_images:
            print(f"\nMissing images count: {len(missing_images)}")
            print("First few missing images:")
            for img in missing_images[:5]:
                print(f"- {img}")
                
        if invalid_images:
            print(f"\nInvalid images count: {len(invalid_images)}")
            print("First few invalid images:")
            for img in invalid_images[:5]:
                print(f"- {img}")

        # Validate all images if requested
        if validate_images:
            print("\nValidating all images...")
            self.validate_all_images()

    def validate_all_images(self):
        """
        Validate all images in the dataset by attempting to open them
        """
        invalid_images = []
        for img_path, _ in tqdm(self.samples, desc="Validating images"):
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                    # Try to convert to RGB to catch potential color mode issues
                    img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"\nError validating image {img_path}: {e}")
                invalid_images.append(img_path)

        if invalid_images:
            print(f"\nFound {len(invalid_images)} corrupted or invalid images:")
            for path in invalid_images[:5]:
                print(f"- {path}")
        else:
            print("\nAll images validated successfully!")

    def _create_samples(self):
        """
        Create samples based on the chosen caption strategy
        """
        samples = []
        for img_name, captions in self.image_to_captions.items():
            if not captions:  # Skip images with no valid captions
                continue
                
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Skipping non-existent image: {img_path}")
                continue
            
            if self.caption_strategy == 'all':
                # Create a sample for each caption
                for caption in captions:
                    samples.append((img_path, caption))
            
            elif self.caption_strategy == 'random':
                # Randomly select one caption
                caption = random.choice(captions)
                samples.append((img_path, caption))
            
            elif self.caption_strategy == 'first':
                # Use only the first caption
                samples.append((img_path, captions[0]))
            
            else:
                raise ValueError(f"Unknown caption strategy: {self.caption_strategy}")
        
        return samples

    def get_all_captions_for_image(self, img_path):
        """
        Get all captions for a given image path
        """
        img_name = os.path.basename(img_path)
        return self.image_to_captions.get(img_name, [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        try:
            # Load image
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        # Process text if tokenizer is provided
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
                'img_path': img_path,
                'all_captions': self.get_all_captions_for_image(img_path)
            }
        else:
            return {
                'image': image,
                'caption': caption,
                'img_path': img_path,
                'all_captions': self.get_all_captions_for_image(img_path)
            }

def flickr_collate_fn(batch):
    """
    Custom collate function for Flickr30k dataset
    """
    images = []
    captions = []
    img_paths = []
    all_captions = []
    input_ids = []
    attention_masks = []
    has_tokens = 'input_ids' in batch[0]

    for item in batch:
        images.append(item['image'])
        captions.append(item['caption'])
        img_paths.append(item['img_path'])
        all_captions.append(item['all_captions'])
        
        if has_tokens:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])

    # Stack all gathered data
    batch_out = {
        'image': torch.stack(images) if torch.is_tensor(images[0]) else images,
        'caption': captions,
        'img_path': img_paths,
        'all_captions': all_captions
    }

    if has_tokens:
        batch_out['input_ids'] = torch.stack(input_ids)
        batch_out['attention_mask'] = torch.stack(attention_masks)

    return batch_out



if __name__ == "__main__":
    # Example usage
    img_dir = '/data/dataset/Flicker30k/Images/'
    caption_file = '/data/dataset/Flicker30k/captions.txt'

    # Create dataset with image validation
    dataset = Flickr30kDataset(
        img_dir=img_dir,
        caption_file=caption_file,
        caption_strategy='first',
        validate_images=False  # Enable full validation
    )

    # print("\nTesting first 5 samples:")
    # for i in range(min(5, len(dataset))):
    #     try:
    #         sample = dataset[i]
    #         print(f"\nSample {i+1}:")
    #         print(f"Image Path: {sample['img_path']}")
    #         print(f"Caption: {sample['caption']}")
    #         print(f"Number of alternative captions: {len(sample['all_captions'])}")
    #         print(f"Image Shape: {sample['image'].shape if isinstance(sample['image'], torch.Tensor) else 'N/A'}")
    #     except Exception as e:
    #         print(f"Error processing sample {i}: {e}")
    #
    # # Test loading all samples
    # print("\nTesting all samples...")
    # errors = []
    # for i in tqdm(range(len(dataset))):
    #     try:
    #         sample = dataset[i]
    #     except Exception as e:
    #         errors.append((i, str(e)))
    #
    # if errors:
    #     print(f"\nFound {len(errors)} errors while testing all samples:")
    #     for idx, error in errors[:5]:
    #         print(f"Sample {idx}: {error}")
    # else:
    #     print("\nAll samples loaded successfully!")