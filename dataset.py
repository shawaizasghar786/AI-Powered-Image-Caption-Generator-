import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import os

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab_path):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.data = []
        with open(caption_file, 'r') as f:
            for line in f:
                img, caption = line.strip().split('\t')
                img = img.split('#')[0]
                tokens = caption.lower().split()
                self.data.append((img, tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, tokens = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        image = self.transform(image)
        caption = [self.vocab['<START>']] + [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens] + [self.vocab['<END>']]
        return image, torch.tensor(caption)
