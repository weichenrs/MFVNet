import os
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import rasterio
import torch
from torch.nn.modules.activation import Sigmoid

from operations import *
from modeling.my_hrnet import *
from modeling.my_pspnet import *
from modeling.my_unet import *
from modeling.Align import AlignModule_full
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class Build_endecoder(nn.Module):
    def __init__(self, dataset):
        super(Build_endecoder, self).__init__()

        self.num_attnfeat = 512
        self.num_scale = 3
        
        if dataset == 'gid':
            num_classes = 5
            self.s1_encoder = PSPNet(layers=101, pretrained=False, classes=num_classes)
            self.s2_encoder = UNet(n_channels=4, n_classes=num_classes)
            self.s3_encoder = HighResolutionNet(HRNET_48, num_classes=num_classes)
            self.s1_temp = nn.Sequential(nn.Conv2d(4096, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s2_temp = nn.Sequential(nn.Conv2d(512, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s3_temp = nn.Sequential(nn.Conv2d(720, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
        elif dataset == 'potsdam':
            self.num_class = 6
            self.s1_encoder = PSPNet(layers=101, pretrained=False, classes=self.num_class)
            self.s2_encoder = PSPNet(layers=101, pretrained=False, classes=self.num_class)
            self.s3_encoder = UNet(n_channels=4, n_classes=self.num_class)
            self.s1_temp = nn.Sequential(nn.Conv2d(4096, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s2_temp = nn.Sequential(nn.Conv2d(4096, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s3_temp = nn.Sequential(nn.Conv2d(512, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
        elif dataset == 'wfv':
            num_classes = 3
            self.s1_encoder = HighResolutionNet(HRNET_48, num_classes=num_classes)
            self.s2_encoder = HighResolutionNet(HRNET_48, num_classes=num_classes)
            self.s3_encoder = UNet(n_channels=4, n_classes=num_classes)
            self.s1_temp = nn.Sequential(nn.Conv2d(720, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s2_temp = nn.Sequential(nn.Conv2d(720, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
            self.s3_temp = nn.Sequential(nn.Conv2d(512, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.num_attnfeat),
                                        nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        self.s2_align = AlignModule_full(512, 64)
        self.s3_align = AlignModule_full(512, 64)
        self.all_attn = nn.Sequential(nn.Conv2d(self.num_attnfeat*self.num_scale, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.num_attnfeat),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.num_attnfeat, self.num_attnfeat, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.num_attnfeat),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.num_attnfeat, self.num_scale, kernel_size=1, bias=False))
        
        for n in [self.s1_encoder,self.s2_encoder,self.s3_encoder]:
            for m in n.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, s1, s2, s3):
        with torch.no_grad():
            s1_feat, s1_out = self.s1_encoder(s1)
            s2_feat, s2_out = self.s2_encoder(s2)
            s3_feat, s3_out = self.s3_encoder(s3)

            h, w = s1.size()[2:]
            f_size = (int(h/4), int(w/4))
            s1_feat = F.interpolate(s1_feat, size=f_size, mode='bilinear', align_corners=True)
            s1_out = F.interpolate(s1_out, size=f_size, mode='bilinear', align_corners=True)

            s2_dsprate = 1.5
            s2_sz1 = s2_feat.shape[-1]
            s2_sz2 = s2_out.shape[-1]
            s2_tmp1 = s2_sz1/s2_dsprate
            s2_tmp2 = s2_sz2/s2_dsprate
            s2_topleft1 = np.int( s2_sz1/2-np.ceil(s2_tmp1/2) )
            s2_topleft2 = np.int( s2_sz2/2-np.ceil(s2_tmp2/2) )
            s2_feat_tmp = TF.crop(s2_feat, s2_topleft1, s2_topleft1, int(np.ceil(s2_tmp1)), int(np.ceil(s2_tmp1)) )
            s2_out_tmp = TF.crop(s2_out, s2_topleft2, s2_topleft2, int(np.ceil(s2_tmp2)), int(np.ceil(s2_tmp2)) )
            s2_feat = F.interpolate(s2_feat_tmp, size=f_size, mode='bilinear', align_corners=True)
            s2_out = F.interpolate(s2_out_tmp, size=f_size, mode='bilinear', align_corners=True)

            s3_dsprate = 2
            s3_sz1 = s3_feat.shape[-1]
            s3_sz2 = s3_out.shape[-1]
            s3_tmp1 = s3_sz1/s3_dsprate
            s3_tmp2 = s3_sz2/s3_dsprate
            s3_topleft1 = np.int( s3_sz1/2-np.ceil(s3_tmp1/2) )
            s3_topleft2 = np.int( s3_sz2/2-np.ceil(s3_tmp2/2) )
            s3_feat_tmp = TF.crop(s3_feat, s3_topleft1, s3_topleft1, int(np.ceil(s3_tmp1)), int(np.ceil(s3_tmp1)) )
            s3_out_tmp = TF.crop(s3_out, s3_topleft2, s3_topleft2, int(np.ceil(s3_tmp2)), int(np.ceil(s3_tmp2)) )
            s3_feat = F.interpolate(s3_feat_tmp, size=f_size, mode='bilinear', align_corners=True)
            s3_out = F.interpolate(s3_out_tmp, size=f_size, mode='bilinear', align_corners=True)

        s1_feat = self.s1_temp(s1_feat)
        s2_feat = self.s2_temp(s2_feat)
        s3_feat = self.s3_temp(s3_feat)
        s2_out = self.s2_align(s1_feat, s2_feat, s2_out)
        s3_out = self.s3_align(s1_feat, s3_feat, s3_out)
        all_feat = torch.cat((s1_feat, s2_feat, s3_feat), dim = 1)
        all_attn = self.all_attn(all_feat)

        final_out = (all_attn)[:,0:1] *(s1_out) +(all_attn)[:,1:2] *(s2_out) + (all_attn)[:,2:] *(s3_out)

        return F.interpolate(final_out, size=s1.size()[2:], mode='bilinear', align_corners=True)