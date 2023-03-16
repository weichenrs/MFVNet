import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignModule_full(nn.Module):
    def __init__(self, in_channels, tmp_channels):
        super(AlignModule_full, self).__init__()
        self.down_t1 = nn.Conv2d(in_channels, tmp_channels, 1, bias=False)
        self.down_t2 = nn.Conv2d(in_channels, tmp_channels, 1, bias=False)
        self.flow_make = nn.Sequential(nn.Conv2d(tmp_channels * 2,tmp_channels,kernel_size=3,padding=1,bias=False),
                                       nn.BatchNorm2d(tmp_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(tmp_channels, 2, kernel_size=3, padding=1, bias=False))

    def forward(self,t1_feature,t2_feature,t2_pred):
        h, w = t1_feature.size()[2:]
        size = (h, w)
        t1_feature = self.down_t1(t1_feature)
        t2_feature = self.down_t2(t2_feature)
        if t1_feature.size() != t2_feature.size():
            t2_feature = F.interpolate(t2_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([t1_feature, t2_feature], 1))
        newt2pred = self.flow_warp(t2_pred, flow, size=size)

        return newt2pred

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h_o, w_o = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*c, 1, 1, 1).type_as(input).to(input.device)

        flow_r = flow.repeat(c, 1, 1, 1).type_as(input).to(input.device)

        grid = grid + flow_r.permute(0, 2, 3, 1) / norm 
        input_r = input.reshape(n*c, 1, h_o, w_o)
        output = F.grid_sample(input_r, grid)
        output = output.reshape(n,c,out_h,out_w)
        return output