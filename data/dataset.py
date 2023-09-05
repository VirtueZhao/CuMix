import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from random import shuffle
import random
import torch.distributed as dist


def read_split_line(line):
    path, class_id = line.split(' ')
    class_name = path.split('/')[1]
    return path, class_name, int(class_id)


class DomainDataset(data.Dataset):
    def __init__(self, data_root, domains, attributes='w2v_domainnet.npy', train=True, validation=False, transformer=None):
        self.domains = domains
        self.n_doms = len(domains)
        self.class_embedding = attributes
        self.data_root = data_root
        self.train = train
        self.val = validation

        if self.val:
            seen_list = os.path.join(self.data_root, "train_classes.npy")
            self.seen_list = list(np.load(seen_list))
            unseen_list = os.path.join(self.data_root, "val_classes.npy")
            self.unseen_list = list(np.load(unseen_list))
        else:
            seen_list_train = os.path.join(self.data_root, "train_classes.npy")
            self.seen_list = list(np.load(seen_list_train))
            seen_list_val = os.path.join(self.data_root, "val_classes.npy")
            self.seen_list = self.seen_list + list(np.load(seen_list_val))
            unseen_list = os.path.join(self.data_root, "test_classes.npy")
            self.unseen_list = list(np.load(unseen_list))

        self.full_classes = self.seen_list + self.unseen_list
        self.seen = torch.LongTensor([self.full_classes.index(k) for k in self.seen_list])
        self.unseen = torch.LongTensor([self.full_classes.index(k) for k in self.unseen_list])
        self.full_classes_idx = torch.cat([self.seen, self.unseen], dim=0)

        if self.train:
            self.valid_classes = self.seen_list
        else:
            self.valid_classes = self.unseen_list

        attributes_list = os.path.join(self.data_root, self.class_embedding)
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

        if isinstance(domains, list):
            for i, d in enumerate(domains):
                self.read_single_domain(d, id=i)
        else:
            self.read_single_domain(domains)

        self.labels = torch.LongTensor(self.labels)
        self.domain_id = torch.LongTensor(self.domain_id)
        self.attributes = torch.cat(self.attributes, dim=0)
        self.classes = len(self.valid_classes)
        self.full_attributes = self.attributes_dict

    def read_single_domain(self, domain, id=0):
        if self.val or self.train:
            file_names = [domain + "_train.txt"]
        else:
            # Note: if we are testing, we use all images of unseen classes contained in the domain,
            # no matter of the split. The images of the unseen classes are NOT present in the training phase.
            file_names = [domain + "_train.txt", domain + "_test.txt"]

        for file_name in file_names:
            self.read_single_file(file_name, id)

    def read_single_file(self, filename, id):
        domain_images_list_path = os.path.join(self.data_root, filename)
        with open(domain_images_list_path, 'r') as files_list:
            for line in files_list:
                line = line.strip()
                local_path, class_name, class_id = read_split_line(line)

                if class_name in self.valid_classes:
                    self.image_paths.append(os.path.join(self.data_root, local_path))
                    self.labels.append(self.valid_classes.index(class_name))
                    self.domain_id.append(id)
                    self.attributes.append(self.attributes_dict[class_name].unsqueeze(0))

    def get_domains(self):
        return self.domain_id, self.n_doms

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        features = self.loader(self.image_paths[index])
        features = self.transformer(features)
        return features, self.attributes[index], self.domain_id[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)


class DistributedBalancedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, samples_per_domain, num_replicas=None, rank=None, shuffle=True, iters='min',
                 domains_per_batch=5):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.domain_ids, self.n_doms = self.dataset.get_domains()
        self.domain_ids = np.array(self.domain_ids)
        self.dict_domains = {}
        self.indices = {}

        


