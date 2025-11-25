import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import json
from PIL import Image
from tqdm import tqdm
import random

class StatefulDefenseDataset(torch.utils.data.Dataset):
    def __init__(self, name=None, transform=None, size=None, start_idx=0, seed=42):
        json_path = './data/{}/{}.json'.format(name, name)
        images_json = list(json.load(open(json_path)).items())
        self.transform = transform
        self.name = name
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(images_json)

        if size:
            images_json = images_json[start_idx:start_idx + size]
        self.data = []
        for img, label in tqdm(images_json, desc="Loading images"):
            img_path = './data/{}/{}'.format(name, img)
            print(img_path)
            print(label)
            self.data.append((img_path, torch.tensor(label)))

        # Targeted attack images, i.e., one image per class
        try:
            self.targeted_dict = json.load(
                open('./data/{}/{}_targeted.json'.format(name, name)))
            self.targeted_dict = {int(k): v for k, v in self.targeted_dict.items()}
            for k, v in self.targeted_dict.items():
                img_path = './data/{}/{}'.format(name, v[0])
                if name == 'imagenet':
                    self.targeted_dict[k] = transform(Image.open(img_path).convert("RGB"))
                else:
                    self.targeted_dict[k] = transform(Image.open(img_path))
        except:
            pass

    def initialize_targeted(self, label):
        return self.targeted_dict[label].unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        if self.name == 'imagenet':
            print(image_path)
            image = self.transform(Image.open(image_path).convert("RGB"))
        else:
            image = self.transform(Image.open(image_path))
        label = self.data[idx][1]
        sample = (image, label, image_path)
        return sample
