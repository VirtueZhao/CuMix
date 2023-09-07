import os
import copy
import json
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
from data.dataset import DomainDataset
from methods import CuMix
from tqdm import tqdm
import random
import numpy as np
from utils import test
import pickle

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DNET_DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

parser = argparse.ArgumentParser()
parser.add_argument("--target", default="clipart")
parser.add_argument("--data_root", default="/dataset/", type=str)
parser.add_argument("--runs", default=2, type=int)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--ckpt_dir", default="./checkpoints", type=str)
parser.add_argument("--config_file", default=None)

args = parser.parse_args()

device_id = 0
torch.cuda.set_device(device_id)

config_file = args.config_file
with open(config_file) as json_file:
    configs = json.load(json_file)
print("Config File: {}".format(config_file))
# print(configs)

multi_domain = True
input_dim = 2048
configs['freeze_bn'] = False
semantic_w = 1.0

DOMAINS = DNET_DOMAINS
target = args.target
sources = copy.deepcopy(DNET_DOMAINS)
sources.remove(target)
dataset = DomainDataset

configs['input_dim'] = input_dim
configs['semantic_w'] = semantic_w
configs['multi_domain'] = multi_domain

log_dir = args.log_dir
checkpoint_dir = args.ckpt_dir
cudnn.benchmark = True
exp_name = "model.pkl"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

log_file = os.path.join(log_dir, (time.strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
checkpoint_file = os.path.join(checkpoint_dir, time.strftime("%Y-%m-%d-%H-%M-%S") + "-runN.pth")

logger = {"results": [], "config": configs, "target": target, "checkpoints": [],
          "sem_loss": [[] for _ in range(args.runs)], "mimg_loss": [[] for _ in range(args.runs)],
          "mfeat_loss": [[] for _ in range(args.runs)]}

results = []
results_top = []
val_datasets = None

train_dataset = dataset(args.data_root, sources, train=True)
test_dataset = dataset(args.data_root, target, train=False)

attributes = train_dataset.full_attributes
seen = train_dataset.seen
unseen = train_dataset.unseen

method = CuMix(seen_classes=seen, unseen_classes=unseen, attributes=attributes, configs=configs)

temp_results = []
top_sources = 0
top_idx = -1

configs['epochs'] = 5

for e in tqdm(range(0, configs['epochs'])):
    semantic_loss, mimg_loss, mfeat_loss = method.fit(train_dataset)
    print("Semantic Loss: {}, Mixup Image Loss: {}, Mixup Feature Loss: {}".format(semantic_loss, mimg_loss, mfeat_loss))
    accuracy = test(method, test_dataset)
    temp_results = accuracy

    logger['sem_loss'].append(semantic_loss)
    logger['mimg_loss'].append(mimg_loss)
    logger['mfeat_loss'].append(mfeat_loss)

checkpoint_dict = {}
method.save(checkpoint_dict)
current_checkpoint_name = checkpoint_file.replace('runN.pth', 'epoch' + str(e+1) + '.pth')
torch.save(checkpoint_dict, current_checkpoint_name)

logger['results'].append(temp_results)
logger['checkpoints'].append(current_checkpoint_name)
print(target, logger['results'][top_idx])

print('\nResults for ' + target, np.mean(logger['results']), np.std(logger['results']))
with open(log_file, 'wb') as handle:
    pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)
