import torch
from torch.utils.data import DataLoader


def test(model, test_data, device='cuda:3', zsl=True):
    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=8, drop_last=False)

    classes = test_data.classes
    target_classes = torch.arange(classes)
    per_class_hits = torch.FloatTensor(classes).fill_(0).to(device)
    per_class_samples = torch.FloatTensor(classes).fill_(0).to(device)

    with torch.no_grad():
        for i, (input, feature_attributes, domains, labels) in enumerate(test_dataloader):
            output = model.predict(input.to(device))
            _, predicted_labels = torch.max(output.data, 1)

            for tgt in target_classes:
                idx = (labels == tgt)
                if idx.float().sum() == 0:
                    continue
                print(tgt)
                print(idx.float().sum())
                print()
                # per_class_hits[tgt] += torch.sum(labels[idx] == predicted_labels[idx].cpu())
                print(labels[idx])
                print(predicted_labels[idx])
            exit()



    exit()







