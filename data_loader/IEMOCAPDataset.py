import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import copy
from tqdm import tqdm


class IEMOCAPDataset(Dataset):
    def __init__(self, data_path, txt_path):
        self.data_path = data_path
        with open(txt_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        line = self.data[idx].strip().split(' [split|sign] ')
        id = line[0]
        caption = line[1]

        video_path = os.path.join(self.data_path, f"{id}.mp4")
        audio_path = os.path.join(self.data_path, f"{id}.wav")

        sample = {
            'video_path': video_path,
            'audio_path': audio_path,
            'caption': caption
        }

        return sample

def IEMOCAP_collate_fn(batch):
    captions = [item['caption'] for item in batch]
    video_paths = [item['video_path'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    batch = {
        'caption': captions,
        'video_path': video_paths,
        'audio_path': audio_paths
    }
    return batch

def load_category(txt_path):
    classes = ["Neutral", "Angry", "Sad", "Happy"]

    labels = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' [split|sign] ')
            label_str = parts[2]
            label_one_hot = torch.zeros(4)
            label_idx = classes.index(label_str)
            label_one_hot[label_idx] = 1
            labels.append(label_one_hot)
    return torch.stack(labels, dim=0)

if __name__ == '__main__':
    dataset = IEMOCAPDataset(data_path='/data2/kudret/data/dataset/IEMOCAP/train',
                            txt_path='/data2/kudret/data/dataset/IEMOCAP/train.txt')
    print(len(dataset))
    print(dataset[0])

    labels = load_category('/data2/kudret/data/dataset/IEMOCAP/train.txt')
    print(len(labels))
    print(labels[:10])