

import os
import warnings
import logging
import logging.handlers
import multiprocessing
import random
import json
import socket
from datetime import datetime
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
from attacks.attacks import *
from attacks.adaptive.Square import Square
from models.statefuldefense import init_stateful_classifier
from models.statefuldefense import AdvQDet
from utils import datasets
import torch.nn as nn
import torchvision.datasets as dset
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import sys
import contextlib
import pandas as pd
from PIL import Image
from utils import datasets
import torchvision.datasets as dset
import torch.utils.data
from torch.cuda.amp import autocast

def p_selection(p_init, step, max_steps=10000):
    # Normalize step to a 0–1 range and scale to the original 10,000-based logic
    scaled_step = int((step / max_steps) * 10000)

    conditions = [
        (10 < scaled_step <= 50, p_init / 2),
        (50 < scaled_step <= 200, p_init / 4),
        (200 < scaled_step <= 500, p_init / 8),
        (500 < scaled_step <= 1000, p_init / 16),
        (1000 < scaled_step <= 2000, p_init / 32),
        (2000 < scaled_step <= 4000, p_init / 64),
        (4000 < scaled_step <= 6000, p_init / 128),
        (6000 < scaled_step <= 8000, p_init / 256),
        (8000 < scaled_step <= 10000, p_init / 512),
    ]


    valid_choices = [p for condition, p in conditions if condition]
    p = random.choice(valid_choices) if valid_choices else p_init
    return p



def add_squares(x, x_adv, s, num_squares, eps, randomeps=False):
    x_adv_candidate = x_adv.clone()
    for _ in range(num_squares):
        pert = x_adv_candidate - x

        center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
        center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
        x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
        x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]

        while torch.sum(
                torch.abs(
                    torch.clamp(
                        x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
                    ) -
                    x_adv_window)
                < 10 ** -7) == x.shape[1] * s * s:
            if(randomeps==False):
                pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(np.random.choice([-eps, eps], size=[x.shape[1], 1, 1])).float().to(x.device)
            else:
                pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(np.random.choice([-eps, eps], size=[x.shape[1], s, s])).float().to(x.device)
        
        import random
        
        x_adv_candidate = torch.clamp(x + (pert), 0, 1)
    return x_adv_candidate

def square_attack_perturb(x, x_adv_candidate, eps, eps2, t, num_of_squares):
    
        #change to p_selection(0.8, t) and num_squares to 1 to simulate standard square attack
        dim = torch.prod(torch.tensor(x.shape[1:]))
     
        s = int(
            min(max(torch.sqrt(p_selection(0.1, t) * dim / x.shape[1]).round().item(), 1),
                x.shape[2] - 1))
        
        
        x_adv_candidate = add_squares(x,x_adv_candidate, s, num_squares=20, eps=eps2, randomeps=False)
        
        
        s = int(
            min(max(torch.sqrt(p_selection(0.0075, t) * dim / x.shape[1]).round().item(), 1),
                x.shape[2] - 1))

        x_adv_candidate = add_squares(x, x_adv_candidate, s, num_squares=150, eps=eps2, randomeps=False)
        
       
        
        
        return x_adv_candidate



CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
TARGET_SIZE = (224, 224)  # (W, H)
