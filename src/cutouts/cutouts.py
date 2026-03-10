import dbof.dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems
from dask.distributed import Client
import numpy as np
from sklearn.model_selection import train_test_split
import torch 


# mean = [0.18799053, 33.53894686, 15.24299731]
# std = [0.68821116, 5.4085488, 9.07874918]

mean = torch.tensor([0.18799053, 33.53894686, 15.24299731], dtype=torch.float32).view(3,1,1)
std  = torch.tensor([0.68821116, 5.4085488, 9.07874918], dtype=torch.float32).view(3,1,1)

def chunk_aware_subsample(da, num_sample_chunks, subsample_per_chunk):
    chunk = 1020
    rng = np.random.default_rng()
    
    n = da.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    
    
    sample_chunks = rng.choice(n_chunks, size=num_sample_chunks, replace=False)
    
    # within each chosen chunk, pick r indices
    
    idx = []
    for c in sample_chunks:
        start = c * chunk
        stop = min((c + 1) * chunk, n)
        idx.append(rng.integers(start, stop, size=subsample_per_chunk))
    
    idx = np.sort(np.concatenate(idx))
    return idx

def make_regime_labels(b, n_classes=3, eps=1e-12):
    """
    grad_b2: array of shape [N, 1, H, W] or [N, H, W]
    Returns:
        patch_stat: [N] median log(grad_b2)
        labels:     [N] integer class labels
        bins:       quantile bin edges
    """

    # one scalar per patch
    patch_stat = np.mean(b, axis=(1, 2))   # [N]

    # quantile bins for balanced classes
    bins = np.quantile(patch_stat, np.linspace(0, 1, n_classes + 1))

    # avoid edge issue at the max
    bins[-1] += 1e-9

    labels = np.digitize(patch_stat, bins[1:-1], right=False)

    return patch_stat, labels.astype(np.int64), bins

from torch.utils.data import Dataset, DataLoader

class OceanPatchDataset(Dataset):
    def __init__(self, X, labels):
        """
        X: array/tensor of shape [N, C, H, W]
        labels: array/tensor of shape [N]
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        
        if len(self.X) != len(self.labels):
            raise ValueError(f"X and labels must have same length, got {len(self.X)} and {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x_norm = (self.X[idx] - mean) / (std + 1e-6)
        return {
            "image": x_norm,
            "label": self.labels[idx],
        }


def make_ocean_dataloaders(X_train, y_train, X_val, y_val, batch_size=64, num_workers=0):
    train_ds = OceanPatchDataset(X_train, y_train)
    val_ds = OceanPatchDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
    
    
def get_cutout_loaders(): 
    # follow link for a progress bar in following steps
    client = Client(n_workers=8)
    
    port = client.scheduler_info()["services"]["dashboard"]
    # For nrp link is :
    #https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status
    print(f"url : https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status")
    
    bucket = "dbof" #data_cfg["bucket"]
    folder = "native_grid_dbof_training_data"
    s3_endpoint = "https://s3-west.nrp-nautilus.io"
    feature_channels = ['Eta', 'Salt', 'Theta', 'U', 'V', 'W', 'relative_vorticity', 'log_gradb']
    run_id = "big_run_00"
    
    fs, fs_synch = filesystems.create_s3_filesystems(s3_endpoint)
    
    reader = zarr_dataset.ZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name="dataset_creation.zarr",
        fs=fs
    )
    
    images_da, ids_da, valid_mask_da = reader.full_dataset_as_dask()
    
    subset = True 
    
    
    subsample_per_chunk = 300
    num_sample_chunks = 30
    if subset:
        N = len(images_da)
        subset_idxs = chunk_aware_subsample(images_da, num_sample_chunks, subsample_per_chunk)
        images_da = images_da[subset_idxs]
        ids_da = ids_da[subset_idxs]
    
    images_np = images_da.compute() 
    
    theta = images_np[:, 2]   # (N, 64, 64)
    
    print("Any NaN:", np.isnan(theta).any())
    print("Any +inf:", np.isposinf(theta).any())
    print("Any -inf:", np.isneginf(theta).any())
    
    print("Total NaNs:", np.isnan(theta).sum())
    print("Total infs:", np.isinf(theta).sum())
    
    # Boolean mask: True if patch has any NaN
    bad_patch_mask = np.isnan(theta).reshape(theta.shape[0], -1).any(axis=1)
    
    num_bad = bad_patch_mask.sum()
    print("Number of patches containing NaN:", num_bad)
    
    bad_indices = np.where(bad_patch_mask)[0]
    print("First 10 bad patch indices:", bad_indices[:10])
    
    N = images_np.shape[0]
    
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[bad_indices] = False
    
    images_clean_np = images_np[keep_mask]
    
    print("Old shape:", images_np.shape)
    print("New shape:", images_clean_np.shape)
    
    vort = images_clean_np[:, 6]   # (N, 64, 64)
    
    print("Any NaN:", np.isnan(vort).any())
    print("Any +inf:", np.isposinf(vort).any())
    print("Any -inf:", np.isneginf(vort).any())
    
    print("Total NaNs:", np.isnan(vort).sum())
    print("Total infs:", np.isinf(vort).sum())
    
    # Boolean mask: True if patch has any NaN
    bad_patch_mask = np.isnan(vort).reshape(vort.shape[0], -1).any(axis=1)
    
    num_bad = bad_patch_mask.sum()
    print("Number of patches containing NaN:", num_bad)
    
    bad_indices = np.where(bad_patch_mask)[0]
    print("First 10 bad patch indices:", bad_indices[:10])
    
    #Filter out cutouts containing nan gradients 
    N = images_clean_np.shape[0]
    
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[bad_indices] = False
    
    images_clean_np = images_clean_np[keep_mask]
    
    B = images_clean_np[:,7]
    B.shape
    
    eta_theta_salt = images_clean_np[:,0:3]
    
    patch_stat, labels, bins = make_regime_labels(B, n_classes=3)

    
    X_train, X_val, y_train, y_val = train_test_split(
        eta_theta_salt, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    ocean_loader, ocean_loader_val = make_ocean_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )
    
    return ocean_loader, ocean_loader_val
    