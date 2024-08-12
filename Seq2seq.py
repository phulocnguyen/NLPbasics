import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate

dataset = datasets.load_dataset("bentrevett/multi30k")

print(dataset)

print(123)