import os, random
import numpy as np
from PIL import Image
from mypath import Path
from torch.utils import data
import rasterio
import torch
import torchvision.transforms.functional as TF

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

class pdSegmentation(data.Dataset):
    NUM_CLASSES = 6

    def __init__(self, args, root=Path.db_root_dir('potsdam', '512'), split="train"):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.3394, 0.3629, 0.3369, 0.3829)
        self.std = (0.1393, 0.1377, 0.1431, 0.1388)
        self.images_base = os.path.join(self.root, self.split, 'image')
        self.annotations_base = os.path.join(self.root, self.split, 'label')
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.tif')
        # self.class_names = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        print("Found %d %s images" % (len(self.files[split]), split))

    def get_pd_labels(self):
        return np.array([
            [255,0,0],    #clutter
            [255,255,255],  #imprevious
            [255,255,0],  #car
            [0,255,0],   #tree
            [0,255,255],  #low vegetation
            [0,0,255] ])  #building
    
    def encode_segmap(self, mask):
        mask = np.uint8(mask)
        label_mask = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(self.get_pd_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = np.uint8(label_mask)
        return label_mask

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        s1_path = self.files[self.split][index].rstrip()
        s1msk_path = s1_path.replace('image','label').replace('.tif','_label.tif')
        name = os.path.basename(s1_path)

        with rasterio.open(s1_path) as s1_img:
            s1_img = s1_img.read().astype(np.float32).transpose(1,2,0)
        s1_img /= 255.0
        s1_img -= self.mean
        s1_img /= self.std
        s1_img = s1_img.transpose(2,0,1)
        s1_img = torch.from_numpy(s1_img).float()

        with rasterio.open(s1msk_path) as s1_msk:
            s1_msk = s1_msk.read()
        s1_msk = np.array(s1_msk).astype(np.uint8).transpose(1,2,0)
        s1_msk = self.encode_segmap(s1_msk).astype(np.float32)
        s1_msk = torch.from_numpy(s1_msk).float()

        s2_path = s1_path.replace('/512','/768').replace('scale2','scale3')
        s3_path = s1_path.replace('/512','/1024').replace('scale2','scale4')
        s4_path = s1_path.replace('/512','/1280').replace('scale2','scale5')

        with rasterio.open(s2_path) as s2_img:
            s2_img = s2_img.read().astype(np.float32).transpose(1,2,0)
        s2_img /= 255.0
        s2_img -= self.mean
        s2_img /= self.std
        s2_img = s2_img.transpose(2,0,1)
        s2_img = torch.from_numpy(s2_img).float()
        
        with rasterio.open(s3_path) as s3_img:
            s3_img = s3_img.read().astype(np.float32).transpose(1,2,0)
        s3_img /= 255.0
        s3_img -= self.mean
        s3_img /= self.std
        s3_img = s3_img.transpose(2,0,1)
        s3_img = torch.from_numpy(s3_img).float()

        with rasterio.open(s4_path) as s4_img:
            s4_img = s4_img.read().astype(np.float32).transpose(1,2,0)
        s4_img /= 255.0
        s4_img -= self.mean
        s4_img /= self.std
        s4_img = s4_img.transpose(2,0,1)
        s4_img = torch.from_numpy(s4_img).float()

        if 'train' in self.split:
            a = random.random()
            if a < 0.5:
                s1_img = TF.hflip(s1_img)
                s1_msk = TF.hflip(s1_msk)
                s2_img = TF.hflip(s2_img)
                s3_img = TF.hflip(s3_img)
                s4_img = TF.hflip(s4_img)
            b = random.random()
            if b < 0.5:
                s1_img = TF.vflip(s1_img)
                s1_msk = TF.vflip(s1_msk)
                s2_img = TF.vflip(s2_img)
                s3_img = TF.vflip(s3_img)
                s4_img = TF.vflip(s4_img)
            c = random.random()
            if c < 0.5:
                s1_img = TF.rotate(s1_img, 90)
                s1_msk = TF.rotate(s1_msk.unsqueeze(0), 90).squeeze()
                s2_img = TF.rotate(s2_img, 90)
                s3_img = TF.rotate(s3_img, 90)
                s4_img = TF.rotate(s4_img, 90)

        sample = {'s1_img': s1_img, 's1_msk': s1_msk, 'name': name, 's2_img': s2_img, 's3_img': s3_img, 's4_img': s4_img}
        return sample