import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


class DomainDataset(data.Dataset):
    def __init__(self, data_root, domains, attributes='w2v_domainnet.npy', train=True, validation=False, transformer=None):
        self.domains = domains
        self.n_doms = len(domains)
        self.class_embedding = attributes
        self.data_root = data_root
        self.train = train
        self.val = validation

        if self.val:
            seen_list = os.path.join(self.data_root + "/dataset", "train_classes.npy")
            self.seen_list = list(np.load(seen_list))
            unseen_list = os.path.join(self.data_root + "/dataset", "val_classes.npy")
            self.unseen_list = list(np.load(unseen_list))
        else:
            seen_list_train = os.path.join(self.data_root + "/dataset", "train_classes.npy")
            self.seen_list = list(np.load(seen_list_train))
            seen_list_val = os.path.join(self.data_root + "/dataset", "val_classes.npy")
            self.seen_list = self.seen_list + list(np.load(seen_list_val))
            unseen_list = os.path.join(self.data_root + "/dataset", "test_classes.npy")
            self.unseen_list = list(np.load(unseen_list))

        self.full_classes = self.seen_list + self.unseen_list
        self.seen = torch.LongTensor([self.full_classes.index(k) for k in self.seen_list])
        self.unseen = torch.LongTensor([self.full_classes.index(k) for k in self.unseen_list])
        self.full_classes_idx = torch.cat([self.seen, self.unseen], dim=0)

        if self.train:
            self.valid_classes = self.seen_list
        else:
            self.valid_classes = self.unseen_list

        attributes_list = os.path.join(self.data_root + "/dataset", self.class_embedding)
        self.attributes_dict = np.load(attributes_list, allow_pickle=True, encoding="latin1").item()

        for key in self.attributes_dict.keys():
            self.attributes_dict[key] = torch.from_numpy(self.attributes_dict[key]).float()

        for i, k in enumerate(self.full_classes):
            self.attributes_dict[i] = self.attributes_dict[k]

        self.image_paths = []
        self.labels = []
        self.attributes = []
        self.domain_id = []

        self.loader = default_loader

        if transformer is None:
            self.transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])
        else:
            self.transformer = transformer

        # if isinstance(domains, list):
        #     for i, d in enumerate(domains):
        #         self.read_single_domain(d, id=i)
        # else:
        #     self.read_single_domain(domains)

        self.labels = torch.LongTensor(self.labels)
        self.domain_id = torch.LongTensor(self.domain_id)
        # self.attributes = torch.cat(self.attributes, dim=0)
        self.classes = len(self.valid_classes)
        self.full_attributes = self.attributes_dict

        print(self.labels)
        print(self.domain_id)
        print(self.attributes)
        print(self.classes)
        print(self.full_attributes.keys())



