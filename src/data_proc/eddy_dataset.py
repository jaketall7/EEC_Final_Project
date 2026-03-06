import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_label_csv(path):
    # labels are like: 0,0,0,1,1,2,...
    with open(path, "r") as f:
        txt = f.read().strip()
    if not txt:
        return np.array([], dtype=np.int64)
    return np.array([int(x) for x in txt.split(",")], dtype=np.int64)

def split_scene_ids(images_root, test_size=0.2, seed=42):
    scene_ids = sorted([
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    ])
    train_ids, val_ids = train_test_split(scene_ids, test_size=test_size, random_state=seed)
    return train_ids, val_ids


class EddyPatchDataset(Dataset):
    """
    Builds a patch-level dataset from scene folders.

    Each __getitem__ returns:
        x: FloatTensor (C, patch, patch)
        y: LongTensor scalar (0/1/2)
        meta: dict with scene_id and candidate index (optional)
    """

    def __init__(
        self,
        images_root,         # "train_images_pkl"
        labels_root,         # "LABELS_TRAIN/train"
        scene_ids=None,      # list of scene folder names; if None, use all folders in images_root, this is for val split most likely
        channels=("SST", "CHLA", "sealevel"),  # which .pkl fields to use as channels
        patch=224,
        mean=[17.8596782, 0.26425405, 0.03411109],           # per-channel mean (len C) computed on train scenes only
        std=[7.0410507,  0.57394481, 0.09307475],            # per-channel std (len C)
        return_meta=False
    ):
        self.images_root = images_root
        self.labels_root = labels_root
        self.channels = channels
        self.patch = patch
        self.return_meta = return_meta

        if scene_ids is None:
            # folders directly under images_root
            scene_ids = sorted([
                d for d in os.listdir(images_root)
                if os.path.isdir(os.path.join(images_root, d))
            ])
        self.scene_ids = scene_ids

        C = len(channels)
        self.mean = np.zeros(C, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = np.ones(C, dtype=np.float32)  if std  is None else np.asarray(std,  dtype=np.float32)

        # Build an index of (scene_id, candidate_idx)
        self.index = []
        for sid in self.scene_ids:
            scene_dir = os.path.join(self.images_root, sid)
            label_path = os.path.join(self.labels_root, f"labels_{sid}.csv")
            labels = load_label_csv(label_path)

            n = len(labels)
            for i in range(n):
                self.index.append((sid, i))

        if len(self.index) == 0:
            raise RuntimeError("No samples found. Check folder names and label paths.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, i = self.index[idx]
        scene_dir = os.path.join(self.images_root, sid)

        y = load_label_csv(os.path.join(self.labels_root, f"labels_{sid}.csv"))[i]

        # Build channels
        chans = []
        for c, name in enumerate(self.channels):
            arr = load_pkl(os.path.join(scene_dir, f"{name}.pkl"))

            img = arr[i]

            if isinstance(img, np.ma.MaskedArray):
                img = img.filled(float(0))


            img = np.asarray(img, dtype=np.float32)

            img = torch.tensor(img, dtype=torch.float32)
            img = img.unsqueeze(0)  # (1, H, W)
            resize = Resize((self.patch, self.patch ))
            img = resize(img)
            img = img.squeeze()

            # normalize with train stats (per channel)
            img = (img - self.mean[c]) / (self.std[c] + 1e-6)

            # print(img.shape)
            chans.append(img)

        x = np.stack(chans, axis=0)  # (C, patch, patch)
        x = torch.from_numpy(x).float()
        y = torch.tensor(int(y), dtype=torch.long)

        return { # this matches the hf stuff
            "image": x,
            "label": torch.tensor(int(y), dtype=torch.long),
            "scene_id": sid,
            "candidate_idx": i,
        }
        # if self.return_meta:
        #     return x, y, {"scene_id": sid, "candidate_idx": i}
        # return x, y