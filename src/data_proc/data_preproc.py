from datasets import load_dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

def get_mini_imagnet():

    tiny_imgnet_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    tiny_imgnet_ds_val = load_dataset("zh-plus/tiny-imagenet", split="valid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tiny_imgnet_ds = tiny_imgnet_ds.with_format("torch", device=device)
    tiny_imgnet_ds_val = tiny_imgnet_ds_val.with_format("torch", device=device)

    return tiny_imgnet_ds, tiny_imgnet_ds_val


def preproc_and_normalize_tiny_img_net(tiny_imgnet_ds, tiny_imgnet_ds_val) :
    mean = [0.4914, 0.4822, 0.4465]  # calculated for this dataset
    std = [0.2023, 0.1994, 0.2010]

    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),  # force grey scales to rgb
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # normalize based on data
    ])

    def hf_transform(batch):
        batch["image"] = [transform(im) for im in batch["image"]]
        return batch

    tiny_imgnet_ds = tiny_imgnet_ds.with_transform(hf_transform)
    tiny_imgnet_ds_val = tiny_imgnet_ds_val.with_transform(hf_transform)

    return tiny_imgnet_ds, tiny_imgnet_ds_val

def tiny_image_net_dataloaders():
    tiny_imgnet_ds, tiny_imgnet_ds_val = get_mini_imagnet()
    tiny_imgnet_ds, tiny_imgnet_ds_val = preproc_and_normalize_tiny_img_net(tiny_imgnet_ds, tiny_imgnet_ds_val)

    loader = DataLoader(
        tiny_imgnet_ds,
        batch_size=64,
        shuffle=False
    )

    loader_val = DataLoader(
        tiny_imgnet_ds_val,
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