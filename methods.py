import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class UnitClassifier(nn.Module):
    def __init__(self, attributes, classes, device='cuda'):
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

    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False,
                 device='cuda', world_size=1, rank=0):
        self.end_to_end = True
        self.domain_mix = True

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



