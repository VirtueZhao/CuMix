import os
import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import datetime


RG = np.random.default_rng(seed=42)


def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]


def derange(xs):
    x_new = [] + xs
    for a in range(1, len(x_new)):
        b = RG.choice(range(0, a))
        swap(x_new, a, b)
    return x_new


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions, dim=1), dim=1))
    return loss


def std_mix(x, indices, ratio):
    return ratio * x + (1. - ratio) * x[indices]


class UnitClassifier(nn.Module):
    def __init__(self, attributes, classes, device='cuda:3'):
        super(UnitClassifier, self).__init__()
        self.fc = nn.Linear(attributes[0].size(0), classes.size(0), bias=False).to(device)

        for i, c in enumerate(classes):
            norm_attributes = attributes[c.item()].to(device)
            norm_attributes /= torch.norm(norm_attributes, 2)
            self.fc.weight[i].data[:] = norm_attributes

    def forward(self, x):
        o = self.fc(x)
        return o


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except:
            print('bias not present')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CuMix:

    def __init__(self, seen_classes, unseen_classes, attributes, configs, device='cuda:3', world_size=1, rank=0):
        self.end_to_end = True
        self.domain_mix = True
        self.configs = configs

        backbone = eval(configs['backbone'])
        os.environ['TORCH_HOME'] = '/data/dzha866/Project/CuMix/'
        self.backbone = backbone(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.lr_net = configs['lr_net']
        self.backbone.to(device)
        self.backbone.eval()

        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.attributes = attributes
        self.rank = rank
        self.world_size = world_size
        self.device = device

        attSize = self.attributes[0].size(0)

        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        self.semantic_projector = nn.Linear(input_dim, attSize)
        self.semantic_projector.apply(weights_init)
        self.semantic_projector.to(self.device)
        self.semantic_projector.eval()

        self.train_classifier = UnitClassifier(self.attributes, seen_classes, self.device)
        self.train_classifier.eval()

        self.final_classifier = UnitClassifier(self.attributes, unseen_classes, self.device)
        self.final_classifier.eval()

        self.dpb = configs['domains_per_batch']
        self.iters = configs['iters_per_epoch']

        self.criterion = nn.CrossEntropyLoss()
        self.mixup_criterion = manual_CE
        self.current_epoch = -1

    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()

    def predict(self, input):
        features = self.backbone(input)
        semantic_projection = self.semantic_projector(features)
        prediction = self.final_classifier(semantic_projection)
        return prediction

    def forward(self, input, return_features=False):
        features = self.backbone(input)
        semantic_projection = self.semantic_projector(features)
        prediction = self.train_classifier(semantic_projection)
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self, features):
        semantic_projection = self.semantic_projector(features)
        prediction = self.train_classifier(semantic_projection)
        return prediction

    def get_sample_mixup(self, domains):
        if self.dpb > 1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indices = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indices = torch.randperm(domains.size(0))
        return indices.long()

    def get_ratio_mixup(self, domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    def get_mixup_sample_and_ratio(self, domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    def get_mixed_input_labels(self, input, labels, indices, ratios, dims=2):
        if dims == 4:
            return std_mix(input, indices, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indices, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indices, ratios.unsqueeze(-1)), std_mix(labels, indices, ratios.unsqueeze(-1))

    def fit(self, data):
        self.current_epoch += 1
        self.mixup_beta = min(self.max_beta, max(self.max_beta * self.current_epoch / self.mixup_step, 0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2. - self.current_epoch) / self.mixup_step, 0.0))

        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)

        scale_lr = 0.1 ** (self.current_epoch // self.step)
        optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                  weight_decay=self.decay, nesterov=self.nesterov)
        optimizer_zsl = optim.SGD(self.semantic_projector.parameters(), lr=self.lr * scale_lr, momentum=0.9,
                                  weight_decay=self.decay, nesterov=self.nesterov)

        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.

        for i, (inputs, feature_attributes, domains, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            one_hot_labels = self.create_one_hot(labels)

            preds, features = self.forward(inputs, return_features=True)

            semantic_loss = self.criterion(preds, labels)
            sem_loss += semantic_loss.item()

            mix_indices, mix_ratios = self.get_mixup_sample_and_ratio(domains)
            mix_ratios = mix_ratios.to(inputs.device)
            mixup_features, mixup_labels = self.get_mixed_input_labels(features, one_hot_labels, mix_indices, mix_ratios)

            mixup_features_predictions = self.forward_features(mixup_features)
            mixup_feature_loss = self.mixup_criterion(mixup_features_predictions, mixup_labels)

            total_loss = self.semantic_w * semantic_loss + self.mixup_feat_w * mixup_feature_loss

            mfeat_loss += mixup_feature_loss.item()

            mix_indices, mix_ratios = self.get_mixup_sample_and_ratio(domains)
            mixup_inputs, mixup_labels = self.get_mixed_input_labels(inputs, one_hot_labels, mix_indices,
                                                                     mix_ratios.to(self.device), dims=4)
            mixup_img_predictions = self.forward(mixup_inputs, return_features=False)
            mixup_img_loss = self.mixup_criterion(mixup_img_predictions, mixup_labels)
            total_loss = total_loss + self.mixup_w * mixup_img_loss
            mimg_loss += mixup_img_loss.item()

            self.zero_grad()
            total_loss.backward()
            optimizer_net.step()
            optimizer_zsl.step()

            del total_loss

            if (i + 1) % 20 == 0:
                info = []
                info += [f"epoch [{self.current_epoch + 1}/{self.configs['epochs']}]"]
                info += [f"batch [{i + 1}/{len(dataloader)}]"]
                info += [f"sem_loss {semantic_loss}"]
                info += [f"mimg_loss {mimg_loss}"]
                info += [f"mfeat_loss {mfeat_loss}"]

                print(" ".join(info))

            if (i + 1) == 3:
                self.eval()
                return sem_loss / (i + 1), mimg_loss / (i + 1), mfeat_loss / (i + 1)

        self.eval()

        return sem_loss / (i + 1), mimg_loss / (i + 1), mfeat_loss / (i + 1)
