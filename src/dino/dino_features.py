import random
import torch
from tqdm import tqdm

def extract_data_tiny_image_net(batch):
    return batch["image"], batch["label"]


def append_features(batch, device, model, all_feats, all_labels, data_type):
    if (data_type=="tiny_image_net"):
        images, labels = extract_data_tiny_image_net(batch)

    images = images.to(device, non_blocking=True).float()
    feats = model(images)  # (B, D) CLS embedding
    all_feats.append(feats.cpu())
    all_labels.append(labels.cpu())

def frozen_features(model, loader, device, subsample_percent=None, data_type="tiny_image_net"):
    all_feats = []
    all_labels = []
    print()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):

            if subsample_percent is not None:
                random_integer = random.randint(1, 100)
                if random_integer <= subsample_percent:
                    append_features(batch, device, model, all_feats, all_labels, data_type)
            else :
                append_features(batch, device, model, all_feats, all_labels, data_type)

    X = torch.cat(all_feats, dim=0)
    y = torch.cat(all_labels, dim=0)

    print(X.shape, y.shape)

    return X, y