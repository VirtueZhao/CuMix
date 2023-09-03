import torch.utils.data as data


class DomainDataset(data.Dataset):
    def __init__(self, data_root, domains, attributes='w2v_domainnet.npy', train=True, validation=False, transformers=None):
        self.domains = domains
        self.n_doms = len(domains)
        self.class_embedding = attributes
        self.data_root = data_root
        self.train = train
        self.val = validation
