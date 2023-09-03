import json
import torch
import argparse
from data.dataset import DomainDataset

DNET_DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

parser = argparse.ArgumentParser()
parser.add_argument("--target", default="clipart")
parser.add_argument("--data_root", default="/dataset/", type=str)
parser.add_argument("--runs", default=10, type=int)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--config_file", default=None)

args = parser.parse_args()

device_id = 0
torch.cuda.set_device(device_id)

config_file = args.config_file
with open(config_file) as json_file:
    configs = json.load(json_file)
print("Config File: {}".format(config_file))
# print(configs)

target = args.target
multi_domain = True
DOMAINS = DNET_DOMAINS
dataset = DomainDataset
print(dataset)
