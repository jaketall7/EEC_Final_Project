class SimClrIndexedImageDataset(IndexedImageDataset):

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert("RGB")

        xi = self.transform(image)
        xj = self.transform(image)
        return xi, xj