import os, random
import numpy as np
from PIL import Image
from mypath import Path
from torch.utils import data
import rasterio
import torch
import torchvision.transforms.functional as TF

def get_gid_labels():
    return np.array([
        [255,0,0],    #buildup
        [0,255,0],   #farmland
        [0,255,255],  #forest
        [255,255,0],  #meadow
        [0,0,255] ])  #water

class gidSegmentation(data.Dataset):
    NUM_CLASSES = 5

    def __init__(self, args, root=None, split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.4965, 0.3704, 0.3900, 0.3623)
        self.std = (0.2412, 0.2297, 0.2221, 0.2188)

        self.images_base = os.path.join(self.root, self.split, 'image')
        self.annotations_base = os.path.join(self.root, self.split, 'label')
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')

        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()

        self.class_names = ['buildup', 'farmland', 'forest', 'meadow', 'water']
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
        _tmp = self.encode_segmap(_tmp).astype(np.float32)
        _img /= 255.0
        _img -= self.mean
        _img /= self.std
        _img = _img.transpose(2,0,1)
        _img = torch.from_numpy(_img).float()
        _tmp = torch.from_numpy(_tmp).float()

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

    def encode_segmap(self, mask):
        """Encode segmentation label images as gid classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = np.uint8(mask)
        label_mask = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(get_gid_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = np.uint8(label_mask)
        return label_mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]