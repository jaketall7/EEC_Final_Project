from datasets import load_dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import DatasetDict
#calcuated mean std

mean_erosat = [0.31268863, 0.34511476, 0.37031967]
std_eurosat = [0.19137047, 0.12704164, 0.10665024]

mean_tny_imgnet = [0.4914, 0.4822, 0.4465]
std_tny_imgnet = [0.2023, 0.1994, 0.2010]

mean_merced = [0.48368496, 0.48899795, 0.44999416]
std_merced = [0.21761176, 0.2016371, 0.19559446]

def get_mini_imagnet():
    tiny_imgnet_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    tiny_imgnet_ds_val = load_dataset("zh-plus/tiny-imagenet", split="valid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tiny_imgnet_ds = tiny_imgnet_ds.with_format("torch", device=device)
    tiny_imgnet_ds_val = tiny_imgnet_ds_val.with_format("torch", device=device)

    return tiny_imgnet_ds, tiny_imgnet_ds_val

def get_eurosat_rgb():
    eurosat_rgb_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    eurosat_rbg_ds_val = load_dataset("blanchon/EuroSAT_RGB", split="validation")

    return eurosat_rgb_ds, eurosat_rbg_ds_val

def make_train_val_split(dataset_dict, val_size=0.2, seed=42):
    split = dataset_dict["train"].train_test_split(
        test_size=val_size,
        seed=seed,
        stratify_by_column='label'
    )

    return DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

def get_uc_merced():
    UC_Merced = load_dataset("blanchon/UC_Merced")
    UC_Merced_train_val = make_train_val_split(UC_Merced)
    return UC_Merced_train_val["train"], UC_Merced_train_val["validation"]

def preproc_and_normalize_hf_ds(ds, val, mean, std) :
    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),  # force grey scales to rgb
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # normalize based on data
    ])

    def hf_transform(batch):
        batch["image"] = [transform(im) for im in batch["image"]]
        return batch

    ds = ds.with_transform(hf_transform)
    val = val.with_transform(hf_transform)

    return ds, val

def get_dataloaders(data = "tiny_imagenet"):

    if data == "tiny_imagenet":
        ds, ds_val = get_mini_imagnet()
        ds, ds_val = preproc_and_normalize_hf_ds(ds, ds_val, mean_tny_imgnet, std_tny_imgnet)

    elif data == "eurosat_rgb":
        ds, ds_val = get_eurosat_rgb()
        ds, ds_val = preproc_and_normalize_hf_ds(ds, ds_val, mean_erosat, std_eurosat)

    elif data == "uc_merced":
        ds, ds_val = get_uc_merced()
        ds, ds_val = preproc_and_normalize_hf_ds(ds, ds_val, mean_merced, std_merced)


    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False
    )

    loader_val = DataLoader(
        ds_val,
        batch_size=64,
        shuffle=False
    )

    return loader, loader_val

# # HOW STD AND MEAN WERE CALCULATED:
#
#
# # compute mean and std
# mean = torch.zeros(3)
# std = torch.zeros(3)
# total_images = 0
# print()
# for batch in tqdm(loader, total=len(loader)):
#     images = batch["image"]
#     batch_size = images.size(0)
#
#     # print(batch_size)
#     # break
#     images = images.view(batch_size, 3, -1)
#
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)
#     total_images += batch_size
#
# mean /= total_images
# std /= total_images
#
# print("mean:", mean)
# print("std:", std)