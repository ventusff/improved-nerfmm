from utils import find_files

import imageio
import skimage
from tqdm import tqdm
from skimage.transform import rescale

import torch
from torch.utils.data.dataset import Dataset


class NeRFMMDataset(Dataset):
    def __init__(self, data_dir, downscale=1.):
        super().__init__()
        img_paths = find_files(data_dir)
        imgs = []
        assert len(img_paths) > 0, "no object in the data directory: [{}]".format(data_dir)
        #------------
        # load all imgs into memory
        #------------
        for path in tqdm(img_paths, '=> Loading data...'):
            img = imageio.imread(path)[:, :, :3]
            img = skimage.img_as_float32(img)
            img = rescale(img, 1./downscale, anti_aliasing=True, multichannel=True)
            imgs.append(img)
        self.imgs = imgs
        self.H, self.W, _ = imgs[0].shape
        print("=> dataset: size [{} x {}] for {} images".format(self.H, self.W, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.from_numpy(self.imgs[index]).reshape([-1, 3])
        index = torch.tensor([index]).long()
        return index, img
