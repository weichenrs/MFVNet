import os, random
import numpy as np
from PIL import Image
from mypath import Path
from torch.utils import data
import rasterio
import torch
import torchvision.transforms.functional as TF

class wfvSegmentation(data.Dataset):
    NUM_CLASSES = 3

    def __init__(self, args, root=None, split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.50824948, 0.49074576, 0.44849852, 0.52302321)
        # [524.00521016, 505.95887386, 462.40197871, 539.2369273 ]
        self.std = (0.35359204, 0.34764277, 0.34601062, 0.35922101)
        # [364.55339808, 358.41969806, 356.73695374, 370.35685877]
        self.images_base = os.path.join(self.root, self.split, 'image')
        self.annotations_base = os.path.join(self.root, self.split, 'label')
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')

        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()

        self.class_names = ['clear', 'shadow', 'cloud']
        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path.replace('image','label').replace('.tif','_label.tif')
        name = os.path.basename(img_path)
        with rasterio.open(img_path) as image:
            _img = image.read().astype(np.float32).transpose(1,2,0)
        with rasterio.open(lbl_path) as label:
            _tmp = label.read()
        _tmp = np.array(_tmp).astype(np.uint8).transpose(1,2,0)
        _tmp[_tmp == 0] = 254
        _tmp[_tmp == 1] = 0     # clear
        _tmp[_tmp == 128] = 1   # shadow
        _tmp[_tmp == 255] = 2   # cloud
        _tmp[_tmp == 254] = 255
        _tmp = torch.from_numpy(_tmp).float().squeeze()

        _img /= 1031.0
        _img -= self.mean
        _img /= self.std
        _img = _img.transpose(2,0,1)
        _img = torch.from_numpy(_img).float()

        if 'train' in self.split:
            a = random.random()
            if a < 0.5:
                _img = TF.hflip(_img)
                _tmp = TF.hflip(_tmp)
            b = random.random()
            if b < 0.5:
                _img = TF.vflip(_img)
                _tmp = TF.vflip(_tmp)
            c = random.random()
            if c < 0.5:
                _img = TF.rotate(_img, 90)
                _tmp = TF.rotate(_tmp.unsqueeze(0), 90).squeeze()

        sample = {'image': _img, 'label': _tmp, 'name': name}

        return sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]