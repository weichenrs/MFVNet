import os, random
import numpy as np
from PIL import Image
from mypath import Path
from torch.utils import data
import rasterio
import torch
import torchvision.transforms.functional as TF

def get_potsdam_labels():
    return np.array([
        [255,0,0],    #clutter
        [255,255,255],  #imprevious
        [255,255,0],  #car
        [0,255,0],   #tree
        [0,255,255],  #low vegetation
        [0,0,255] ])  #building

class potsdamSegmentation(data.Dataset):
    NUM_CLASSES = 6

    def __init__(self, args, root=None, split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.3394, 0.3629, 0.3369, 0.3829)
        self.std = (0.1393, 0.1377, 0.1431, 0.1388)
        
        self.images_base = os.path.join(self.root, 'image', self.split)
        self.annotations_base = os.path.join(self.root, 'label', self.split)
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')

        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()
            
        self.class_names = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path.replace('\image','\label').replace('.tif','_label.tif')
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
        img = torch.from_numpy(_img).float()
        mask = torch.from_numpy(_tmp).float()

        if 'train' in self.split:
            a = random.random()
            if a < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            b = random.random()
            if b < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            c = random.random()
            if c < 0.5:
                img = TF.rotate(img, 90)
                mask = TF.rotate(mask.unsqueeze(0), 90).squeeze()
                
        sample = {'image': img, 'label': mask, 'name': name}
        return sample

    def encode_segmap(self, mask):
        """Encode segmentation label images as potsdam classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = np.uint8(mask)
        label_mask = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(get_potsdam_labels()):
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

