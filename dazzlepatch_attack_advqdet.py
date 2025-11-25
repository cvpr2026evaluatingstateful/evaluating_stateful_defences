

# cifar_patch_224_demo.py
import re
from typing import Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms as tvT
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from typing import Optional
from torch.cuda.amp import autocast
import pandas as pd
import contextlib
import sys
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torch.nn as nn
from utils import datasets
from models.statefuldefense import AdvQDet
from models.statefuldefense import init_stateful_classifier
from attacks.adaptive.Square import Square
from attacks.attacks import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
from torchvision.transforms.functional import to_tensor
from argparse import ArgumentParser
import time
from datetime import datetime
import socket
import json
import multiprocessing
import logging.handlers
import logging
import warnings
import os
from multi_square_attack import *
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# image_bank_random_dest_demo.py

# image_bank_unique_dest_demo.py


TARGET_SIZE = (224, 224)  # (W, H)

# load imagenet



class FlatImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.paths = [os.path.join(folder_path, fname)
                      for fname in os.listdir(folder_path)
                      if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]  # or just return img

# ------------------------------
# Build an IMAGE BANK (list of (PIL 224x224, label))
# ------------------------------


def build_image_bank(dataset, bank_size=200):
    bank = []
    labels = [13, 289, 802, 803, 296, 270, 279, 2, 3, 4, 6, 21, 22, 23, 24]
    for i in range(len(dataset)):
        img, label = dataset[i]  # PIL when transform=None

        if not isinstance(img, Image.Image):
            img = tvT.ToPILImage()(img)
        bank.append((img.resize(TARGET_SIZE, Image.BILINEAR), label))
    return bank


# randomly positioning

def add_image_as_patch(
    src_img,
    dst_img,
    patch_size_px,         # square size in pixels (e.g., 56, 96, 128)
    alpha=None,            # None = hard paste; float in [0,1]; or (min,max) -> uses midpoint
    keep_aspect=True,
    target_size=(224, 224)
):
    # Ensure PIL
    if not isinstance(src_img, Image.Image):
        src_img = T.ToPILImage()(src_img)
    if not isinstance(dst_img, Image.Image):
        
        dst_img = T.ToPILImage()(dst_img)

    # Resize both to the working canvas
    src  = src_img.resize(target_size, Image.BILINEAR)
    base = dst_img.resize(target_size, Image.BILINEAR)
 

    W, H = base.size  # e.g., 224x224

    # Determine patch width/height (tw, th)
    s = max(1, min(int(patch_size_px), W, H))
    if keep_aspect:
        sw, sh = src.size
        aspect = sw / sh if sh != 0 else 1.0
        if aspect >= 1.0:
            tw = s
            th = max(1, int(round(tw / aspect)))
        else:
            th = s
            tw = max(1, int(round(th * aspect)))
    else:
        tw = th = s

    tw, th = min(tw, W), min(th, H)

    # Create the patch by resizing the source to (tw, th)
    patch_img = src.resize((tw, th), Image.BILINEAR)

    # ----- CENTER PLACEMENT -----
    left = (W - tw) // 2
    top  = (H - th) // 2

    if alpha is None:
        # Hard paste (no blending)
        out = base.copy()
        out.paste(patch_img, (left, top))
        return out
    else:
        # Blended paste
        if isinstance(alpha, (tuple, list)) and len(alpha) == 2:
            a = float(alpha[0] + alpha[1]) / 2.0   # deterministic midpoint
        else:
            a = float(alpha)
        a = max(0.0, min(1.0, a))

        out = base.convert("RGBA")
        pr = patch_img.convert("RGBA")

        alpha_layer = Image.new("L", pr.size, int(round(255 * a)))
        pr.putalpha(alpha_layer)

        out.paste(pr, (left, top), pr)
        return out.convert("RGB")


def add_image_as_patch_crop(
    src_img,
    dst_img,
    crop_size_px: Union[int, Tuple[int, int]],
    alpha: Optional[Union[float, Tuple[float, float]]] = None,
    target_size: Tuple[int, int] = (224, 224),
    crop_anchor: str = "center",
    # fade factor in [0,1]; 1=fully visible, 0=black
    dst_fade: Optional[float] = 0.5,
    rng: Optional[random.Random] = None
):
    """
    Crop a region from src_img and paste it at the center of dst_img (no resizing).
    - crop_size_px: int for square or (cw, ch) for width/height in pixels (on the source).
    - alpha: Patch transparency. None = hard paste; float in [0,1]; tuple uses midpoint.
    - dst_fade: Fades the entire destination image by blending with black. 1=no fade, 0=black.
    - crop_anchor: 'center' for center crop, 'random' for random crop on source.
    """

    r = rng if rng is not None else random

    # Ensure PIL
    if not isinstance(src_img, Image.Image):
        src_img = tvT.ToPILImage()(src_img)
    if not isinstance(dst_img, Image.Image):
        dst_img = tvT.ToPILImage()(dst_img)

    # Prepare base image
    base = dst_img.resize(target_size, Image.BILINEAR).convert("RGBA")
    W, H = base.size

    # Optionally fade the entire destination image
    if dst_fade is not None:
        fade_factor = max(0.0, min(1.0, dst_fade))
        # Blend with a black background
        black_bg = Image.new("RGBA", base.size, (0, 0, 0, 255))
        base = Image.blend(black_bg, base, fade_factor)

    # Determine crop size on source
    if isinstance(crop_size_px, int):
        cw = ch = max(1, crop_size_px)
    else:
        cw = max(1, int(crop_size_px[0]))
        ch = max(1, int(crop_size_px[1]))

    sw, sh = src_img.size
    cw, ch = min(cw, sw), min(ch, sh)

    # Crop region on source
    if crop_anchor.lower() == "random":
        left_src = r.randint(0, max(0, sw - cw))
        top_src = r.randint(0, max(0, sh - ch))
    else:
        left_src = (sw - cw) // 2
        top_src = (sh - ch) // 2

    right_src = left_src + cw
    bottom_src = top_src + ch
    patch = src_img.crop(
        (left_src, top_src, right_src, bottom_src)).convert("RGBA")

    # Center position on destination
    pw, ph = patch.size
    left_dst = (W - pw) // 2
    top_dst = (H - ph) // 2

    # Paste patch with optional alpha
    if alpha is None:
        base.paste(patch, (left_dst, top_dst))
    else:
        a = (alpha[0] + alpha[1]) / \
            2.0 if isinstance(alpha, (tuple, list)) else float(alpha)
        a = max(0.0, min(1.0, a))
        alpha_layer = Image.new("L", patch.size, int(round(255 * a)))
        patch.putalpha(alpha_layer)
        base.paste(patch, (left_dst, top_dst), patch)

    return base.convert("RGB")


def _to_pil(x):
    return x if isinstance(x, Image.Image) else Image.open(x)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sanitize_filename(name: str) -> str:
    # Replace spaces with underscores and remove anything unsafe
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def show_data(data, label="Image", save_name=None, save_folder="imagenet_patch_rr", save=False):
    plt.figure()
    plt.title(label)
    # plt.imshow(data, cmap="gray")
    plt.imshow(data)
    plt.axis('off')

    if save and save_name and save_folder:
        os.makedirs(save_folder, exist_ok=True)
        safe_name = sanitize_filename(save_name) + ".png"
        full_path = os.path.join(save_folder, safe_name)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"✅ Saved: {full_path}")

    plt.show()
    plt.close()


warnings.filterwarnings("ignore")


with open('imagenet_class_index.json', 'r') as f:
    class_index = json.load(f)

# Create idx_to_label as a list with correct 0-999 ordering
idx_to_label = [class_index[str(i)][1] for i in range(len(class_index))]


# Mean and Std used for normalization
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# Compute clamp ranges for an image with original pixel values in [0, 1]
clamp_min = (0 - mean) / std  # When original pixel = 0
clamp_max = (1 - mean) / std  # When original pixel = 1

# Example: a normalized image tensor with shape (C, H, W)
# Replace with your normalized image
normalized_tensor = torch.randn(3, 224, 224)

# Reshape clamp values for broadcasting (from shape [3] to [3, 1, 1])
clamp_min = clamp_min.view(3, 1, 1).cuda()
clamp_max = clamp_max.view(3, 1, 1).cuda()


# Function to Denormalize
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape for broadcasting
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean  # Reverse normalization


set_seed(42)

classifier_preprocess = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    
    #transforms.RandomRotation(10),
    transforms.Normalize(               # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


"""
# same like classifier_preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
classifier_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(232),   # resize shorter side
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(     # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""

classifier_preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()


class ImageNetSampleDataset(Dataset):
    def __init__(self, root_dir='./data/imagenet/test_images/Sample_1000', json_path="imagenet_class_index.json", transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()

        with open(json_path, 'r') as f:
            raw_mapping = json.load(f)
        self.folder_to_label = {v[0]: int(k) for k, v in raw_mapping.items()}

        self.samples = []
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = self.folder_to_label.get(folder, -1)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def to_numpy_image(x):
    if isinstance(x, Image.Image):
        return np.array(x)  # HxWxC, uint8
    if torch.is_tensor(x):
        # expect CHW in [0,1] or [0,255]; convert to HWC
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        return x.numpy()
    raise TypeError(f"Unsupported type for to_numpy_image: {type(x)}")


def safe_preprocess(img):
    if isinstance(img, np.ndarray) and img.dtype == np.float32:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return classifier_preprocess(img)


config = json.load(
    open("./configs/imagenet/advqdet/square/untargeted/adaptive/config.json"))
print(config)
model_config, attack_config = config["model_config"], config["attack_config"]

# Load model.
model = init_stateful_classifier(model_config)

classifier = model.model.eval()
classifier.to("cuda")
advqdet_model = AdvQDet(model_config["state"], promptFlag=True,
                        sixteen=False, randomizeprompt=False, prompt_lengthh=20)


# Load dataset.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((224, 224)),])


dataset = ImageNetSampleDataset(
    "./data/imagenet/test_images/Sample_1000", "imagenet_class_index.json", transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
pbar = tqdm(test_loader, colour="yellow")


data_dir = "openimages_snow"

train_raw = FlatImageFolder(data_dir, transform=transform)
#train_raw = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=None)

# Config
BANK_SIZE = 10000         # total images in the bank
N_SRC = 4               # how many sources (rows)
# how many destinations per source (no repeats within the row)
DEST_PER_SOURCE = 200
PATCH_SIZES_POOL = [56, 96, 128, 160]  # we’ll sample sizes from here

rng = random.Random(1337)

# Build image bank
image_bank = build_image_bank(
    train_raw, bank_size=BANK_SIZE)  # list of (PIL224, label)

# Safety: you can’t have more unique destinations than BANK_SIZE-1 (excluding the source)
max_unique_dest = max(0, len(image_bank) - 1)
if DEST_PER_SOURCE > max_unique_dest:
    raise ValueError(
        f"DEST_PER_SOURCE={DEST_PER_SOURCE} exceeds available unique destinations "
        f"({max_unique_dest}). Increase BANK_SIZE or lower DEST_PER_SOURCE."
    )

# Choose which bank indices act as sources
src_indices = list(range(min(N_SRC, len(image_bank))))

cache_hits = 0  # so it measures the 'average' hit rate
all_preds = []
all_labels = []
classifier_missclassifications = 0

twofiftyindex = 0
avg_cache_hits = 0
total_iterations = 0


hundred_query_indexes = np.load("random_imagenet_indices.npy")

print(hundred_query_indexes)


# config
T = 700          # number of iterations per source
total_misclassifications = 0
patchattack_irresistant_samples = []
number_of_samples = 0
total_samples_iterated_across_restarts = 0
for i, (x, y) in enumerate(pbar):

    undetected_eot = 0

    set_seed(seed=48)

    is_misclassified = 0
    cache_hits = 0
    advqdet_model.resetCache()

    print('---------------------------------------------------')
    print('Index {}:'.format(i))
    print("Label: ", y)

    src_img224 = x+0
    src_label = y+0

    x = x.cuda()
    y = y.cuda()
    squareattackloss = 10000
    cache_hits = 0

    # Step 1: Detach, convert to NumPy, transpose for image format
    xshow = x.detach().cpu().numpy()
    xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
    #show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label="clean Index"+str(i), save_name="clean Index"+str(i))

    """
    # Step 2: Apply transforms on xshow (each image individually)
    x_transformed = torch.stack([
        classifier_preprocess((img))  # Convert to 0-255, uint8 for PIL
        for img in xshow
    ])
    
    x_transformed = x_transformed.cuda()
    """
    

    """
    #for second classifier prerpcessor the one that resembles resnetv2.weights.tranforms
    x_transformed = torch.stack([safe_preprocess(img) for img in xshow])


    x_transformed = x_transformed.cuda()
    """

    x_transformed = classifier_preprocess(x+0)

    with torch.no_grad():
        outputs = classifier(x_transformed)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    # Step 5: Loop through each item in the batch
    for j in range(len(x_transformed)):
        pred_idx = preds[j].item()
        true_idx = y[j].item()

        pred_class = idx_to_label[pred_idx]
        true_class = idx_to_label[true_idx]

        confidence = probs[j][pred_idx].item()

        print(f"🔍 Predicted: {pred_class} () with confidence {confidence:.2f}")
        print(f"✅ Actual:    {true_class} ()")
        print("---------------")

        # Accumulate results for accuracy
        all_preds.append(pred_idx)
        all_labels.append(true_idx)

    # Initialize adversarial example
    if (pred_class == true_class):
        
        

        if (i not in hundred_query_indexes):
            continue
        

        if (number_of_samples > 199):
            break

        src_idx = 0
        number_of_samples += 1
      
        total_samples_iterated_across_restarts += 1

        # candidate destinations: all bank members except the source
        choices = [k for k in range(len(image_bank)) if k != src_idx]

        # must have at least T unique destinations available
        if len(choices) < T:
            raise ValueError(
                f"Need {T} unique destinations, but only {len(choices)} available "
                f"(bank_size={len(image_bank)} → {len(image_bank)-1} excluding source). "
                "Increase BANK_SIZE or reduce T."
            )
        
        xshow = x.detach().cpu().numpy()
        xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
        #show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label="clean Index"+str(i), save_name="clean Index"+str(i))
        
        
        # precompute a unique, random destination order (no replacement)
        # unique indices for the next T iterations
        dest_order = rng.sample(choices, k=T)
        squareattackloss = 10000
        twofiftyindex = twofiftyindex+1

        src_img224_best = src_img224.clone()

        eps = 0.05
        eps2 = 0.05
        avgloss = 0
        for t, dest_idx in enumerate(dest_order):
            PATCH_SIZES_POOL = []
            PATCH_SIZES_POOL.append(160)

            if (t < T-1):

                if (undetected_eot == 0):
                    if (t < 350):
                        num_of_squares_for_square_perturb = t
                    else:
                        num_of_squares_for_square_perturb = 350

                    src_img224_squared = square_attack_perturb(
                        (src_img224), (src_img224_best), eps, eps, t, num_of_squares_for_square_perturb)
                    src_img224_squared = torch.clamp(
                        src_img224_squared, src_img224 - eps, src_img224 + eps)

                perturbation = src_img224_squared-src_img224
                to_pil = transforms.ToPILImage()
                src_img224_squared_pil = to_pil(
                    src_img224_squared.squeeze(0).cpu())

                # pick the unique destination for this iteration
                dst_img224, dst_label = image_bank[dest_idx]
                size_px = rng.choice(PATCH_SIZES_POOL)

                # make one patched image for this iteration
                choices = [round(x, 1)
                           for x in [ui * 0.1 for ui in range(3, 11)]]
                num = random.choice(choices)
                
                
                x_pil = add_image_as_patch_crop(
                    src_img=src_img224_squared_pil,       # constant source
                    dst_img=dst_img224,       # unique destination for this t
                    crop_size_px=size_px,
                    dst_fade=1.0
                )
                
                #print(to_tensor(dst_img224).shape)
                #show_data(np.reshape(to_tensor(dst_img224).unsqueeze(0).numpy().transpose(0, 2, 3, 1), newshape=(224, 224, 3)), label=str(t)+" not detected", save_name="Iteration"+ str(t)+ " Index"+str(i))
                
                """
                x_pil = add_image_as_patch(
                    src_img=src_img224_squared_pil,       # constant source
                    dst_img=dst_img224,       # unique destination for this t
                    patch_size_px=size_px,
                )
                """

                # convert PIL → tensor (C,H,W) in [0,1], then add batch dim
                x = to_tensor(x_pil).unsqueeze(0)   # shape: (1, 3, 224, 224)

                #x=src_img224_squared+0
            else:
                x = (src_img224_best)

            x = x.cuda()
            # y = torch.Tensor(np.asarray(src_label)).cuda()
            y = torch.as_tensor(src_label, dtype=torch.long, device="cuda")

            # Step 1: Detach, convert to NumPy, transpose for image format
            
            x_adv_candidate = x + 0
            candidate_store = x + 0
            similar = False

            #print("----------------------------------------------------------------")
            #print("Current Index:", i)
            xshow = x_adv_candidate.cpu().numpy()  # Convert the tensor to a NumPy array
            # print(xshow.shape)  # Print the original shap3e for debugging

            # Assuming x_adv_candidate is in shape (batch_size, channels, height, width)
            # Transpose to (batch_size, height, width, channels)
            xshow = xshow.transpose(0, 2, 3, 1)  # Rearrange axes
            # print(xshow.shape)  # Print the new shape for debugging

            # Display the first image after reshaping
            # show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label=str(t)+" not detected", save_name="Iteration"+ str(t)+ " Index"+str(i))

            x_adv_candidate_squeezed = x_adv_candidate.squeeze(0)

            # fix cached prediction and backward graph computation
            if (t == 0):
                prediction = model.model(x_adv_candidate)
                advf = advqdet_model.img_preprocess(
                    x_adv_candidate_squeezed, normalization=True)
                similarity_result = advqdet_model.resultsTopk(
                    advf.squeeze(0), 1)
                # cache_hits=0
                advqdet_model.add(advf.squeeze(0), prediction)

                x_adv = torch.clamp(x_adv_candidate, 0, 1)
                # x_adv_candidate = add_squares(x, x_adv, s, 1, eps=eps)

            else:

                x_adv_square = x_adv_candidate.squeeze(0) + 0
                x_adv_candidate_unchanged = x_adv_candidate + 0

                x_adv = torch.clamp(x_adv_candidate, 0, 1)
                x_adv_candidate_unchanged = x_adv_candidate + 0

                x_adv_candidate_squeezed = x_adv_candidate.squeeze(0)

                original_embed = advqdet_model.getDigest(
                    x_adv_candidate.squeeze(0), requires_grad=False, preprocess=True)

                if (t < T-1):
                    x_adv_candidate = torch.clamp(
                        x_adv_candidate, x - eps, x + eps)
                else:
                    x_adv_candidate = torch.clamp(
                        x_adv_candidate, src_img224.cuda() - eps, src_img224.cuda() + eps)

                x_adv_candidate = torch.clamp(x_adv_candidate, 0, 1)
                denormalized_x_adv = torch.clamp(x_adv_candidate, 0, 1)
                
                """
                #randomized cropping for advqdet
                random_crop = tvT.RandomResizedCrop(size=224,scale=(0.7, 1.0), ratio=(0.75, 1.33))
                
                x_cropped = random_crop(x_adv_candidate)
                
                x_adv_candidate = torch.clamp(x_cropped, 0, 1)
                """
                
                
                """
                xshow = x_cropped.detach().cpu().numpy()
                xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label="cropped Index"+str(i), save_name="cropped Index"+str(i))
                """
                
                

                advf = advqdet_model.img_preprocess(
                    np.reshape(x_adv_candidate.detach().requires_grad_(False).cpu().numpy().transpose(0, 2, 3, 1),
                               newshape=(224, 224, 3)), normalization=False, pil=False).unsqueeze(0)
                perturbed_embed = advqdet_model.getDigest(
                    advf.squeeze(0), requires_grad=False, preprocess=True)

                cosine_similarity = F.cosine_similarity(
                    original_embed, perturbed_embed, dim=0)
                # print("cosine similarity between original embed and advf:", cosine_similarity)

                similarity_result = advqdet_model.resultsTopk(
                    advf.squeeze(0), 1, preprocesss=True)

                if len(similarity_result) > 0:
                    dist, cached_prediction = similarity_result[0]
                    #print("Cosine distance between best historical match and current embedding: ", dist)
                    candidate_store = torch.cat(
                        (candidate_store, x_adv_candidate), dim=0)

                    if dist >= 0.90:

                        advqdet_model.add(advf.squeeze(
                            0), cached_prediction, preprocess=True)
                        # print("Cached Prediction: ", cached_prediction)

                        x_adv_candidate = x_adv_candidate_unchanged + 0
                        cache_hits += 1
                        avg_cache_hits += 1

                        similar = True
                        denormalized_x_adv = torch.clamp(
                            denormalized_x_adv, x - eps, x + eps)
                        xshow = denormalized_x_adv.detach().cpu().numpy()

                        xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                        # show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label=str(t)+" attack detected")

                    else:

                        advqdet_model.add(advf.squeeze(
                            0), prediction, preprocess=True)

                        denormalized_x_adv = torch.clamp(
                            denormalized_x_adv, x - eps, x + eps)

                        xshow = denormalized_x_adv.detach().cpu().numpy()

                        xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                        
                        
                        """
                        if (t == T-1):
                        #if(t % 50 == 0):
                            show_data(np.reshape(xshow[0], newshape=(
                                224, 224, 3)), label=f"Iteration: {t}")
                        """
                        
                        
                        
                        """
                        x_transformed = torch.stack([
                            # Convert to 0-255, uint8 for PIL
                            classifier_preprocess((img))
                            for img in xshow
                        ])
                        x_transformed = x_transformed.cuda()
                        """
                        

                        """
                        #for second classsifier preprocessor
                        x_transformed = torch.stack([safe_preprocess(img) for img in xshow])


                        x_transformed = x_transformed.cuda()
                        """
                        
                        
                        #denormalized_x_adv=torch.clamp(denormalized_x_adv*255, 0, 255)

                        x_transformed=classifier_preprocess(denormalized_x_adv+0)
                        """
                        xshow = x_transformed.detach().cpu().numpy()

                        xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                        
                        if(t==T-1):
                            show_data(np.reshape(xshow[0], newshape=(224, 224, 3)), label=f"Normalized undetected {t}")
                        """
                        
                        
                        #x_transformed = classifier_preprocess(x_transformed+0)

                        with torch.no_grad():
                            x_transformed = x_transformed.cuda()
                            
                    
                            
                            
                            """
                            rotate = transforms.RandomRotation(degrees=10)
                            # rotate = transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0))
                            
                            xshow = denormalized_x_adv.detach().cpu().numpy()
                            xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                            x_transformed = torch.stack([
                                # Convert to 0-255, uint8 for PIL
                                classifier_preprocess((img))
                                for img in xshow
                            ])
                            x_transformed = x_transformed.cuda()
                            
                           
                            

                            y_logit = classifier((x_transformed))  # [0]

                            correct = torch.argmax(y_logit, dim=1)
                            y_logit01 = torch.zeros(y_logit.shape).cuda().scatter(
                                1, correct.unsqueeze(1), 1.0)

                            for i_iter in range(10):
                                xshow = denormalized_x_adv.detach().cpu().numpy()
                                xshow = xshow.transpose(0, 2, 3, 1)  # [B, H, W, C]
                                x_transformed = torch.stack([
                                    # Convert to 0-255, uint8 for PIL
                                    classifier_preprocess((img))
                                    for img in xshow
                                ])
                                x_transformed = x_transformed.cuda()

                                a = classifier((x_transformed))  # [0]
                                #y_logit = y_logit + a
                                correct = torch.argmax(a, dim=1)
                                
                                y_logitt01 = torch.zeros(a.shape,).cuda().scatter(
                                    1, correct.unsqueeze(1), 1.0)
                                y_logit01 = y_logit01 + y_logitt01
                                if(correct==y.view(-1).long()):
                                    y_logit=a
                            
                                    
                            
                        

                            preds = torch.argmax(y_logit01, dim=1)
                            #y_logit = y_logit / 11
                            
                            probs = F.softmax(y_logit, dim=1)
                            outputs = y_logit+0
                            """
                            

                            torch.set_printoptions(sci_mode=False)

                            outputs = classifier(x_transformed)
                            probs = F.softmax(outputs, dim=1)
                            preds = torch.argmax(probs, dim=1)
                            

                            pred_idx = preds[0].item()
                            confidence = probs[0][pred_idx].item()

                            """
                            true_idx = src_label
                            true_logits = outputs[torch.arange(outputs.size(0)), true_idx]

                            # Mask out the true label to find the second highest
                            masked_outputs = outputs.clone()
                            masked_outputs[torch.arange(outputs.size(0)), true_idx] = float('-inf')
                            
                            # Get the second highest logit per sample
                            second_highest_logits, _ = masked_outputs.max(dim=1)
                            
                            # Compute the difference: true - second best
                            logit_diff = true_logits - second_highest_logits
                            """

                            batch_idx = torch.arange(
                                outputs.size(0), device=outputs.device)
                            true_idx = y.view(-1).long()
                            true_logits = outputs[batch_idx, true_idx]
                            masked = outputs.clone()
                            masked[batch_idx, true_idx] = float('-inf')
                            second, _ = masked.max(dim=1)
                            logit_diff = (true_logits - second).item()
                            
                            #print("true logit:", true_logits)
                            #print("second logit: ", second)

                            # loss_value = probs[0][true_idx].item()

                            loss_value = logit_diff

                            #print("current adv loss value:", loss_value)
                            """

                            print(
                                f"🔍 Predicted: {pred_idx} () with confidence {confidence:.2f}")
                            print(f"✅ Actual:    {true_idx} ()")
                            print("---------------")

                            print("Predicted id:", pred_idx)
                            print("true id:", true_idx)
                            """

                            # predicted_class = torch.argmax(outputs, dim=1)
                            if (pred_idx != true_idx):
                                #print("Sample missclassified with index:", i)
                                
                                
                                #if(1):
                                if (t == T-1):
                                
                                    pred_class = idx_to_label[pred_idx]
                                    true_class = idx_to_label[true_idx]

                                    xshow = denormalized_x_adv.detach().cpu().numpy()
                                    xshow = xshow.transpose(
                                        0, 2, 3, 1)  # [B, H, W, C]
                                    """
                                    show_data(np.reshape(xshow[0], newshape=(
                                        224, 224, 3)), label=f'Misclassified {true_class} as {pred_class}, eps={eps}', save_name=f"Misclassified_{str(true_class)}_as_{str(pred_class)}", save=True)
                                    """
                                    total_misclassifications += 1

                                    patchattack_irresistant_samples.append(i)
                                    #np.save("patchattack_irresistant_samples_exp22_size170_e005_firstresizethennormalizationonly1logit_beyondlinf", patchattack_irresistant_samples)
                                    
                                    
                        if (undetected_eot < 3):

                            avgloss += loss_value

                            undetected_eot += 1
                        else:
                            avgloss += loss_value

                            if (avgloss < squareattackloss):
                                squareattackloss = avgloss
                                x_adv_candidate = denormalized_x_adv + 0

                                src_img224_best = src_img224_squared

                            else:
                                x_adv_candidate = x_adv_candidate_unchanged

                            avgloss = 0
                            undetected_eot = 0
            total_iterations = total_iterations+1

        print("total_misclassifications: ", total_misclassifications)
        print("Cache hits: ", cache_hits)
        print("Avg. cache hits:", avg_cache_hits)
        print("Avg. Hit rate:", avg_cache_hits / (total_iterations+1))
        #print("square loss value:", squareattackloss)
        print("Hit rate:", cache_hits / (t+1))
        print("num of samples iterated:", number_of_samples)
        print("total_samples_iterated_across_restarts: ",
              total_samples_iterated_across_restarts)
        print("Attack Success Rate:", total_misclassifications/number_of_samples)

            
