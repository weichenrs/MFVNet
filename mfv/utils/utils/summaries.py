import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.dataloader_utils import decode_seg_map_sequence
import torch.distributed as dist


class TensorboardSummary(object):
    def __init__(self, directory, use_dist=False):
        self.directory = directory
        self.use_dist = use_dist

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            grid_image1 = make_grid(image[:3,:3].clone().cpu().data, 3, normalize=True)

            grid_image2 = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))

            grid_image3 = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            # grid_image4 = make_grid(image[:3,4:].clone().cpu().data, 3, normalize=True)
            # grid_image5 = make_grid(image[:3,3:4].clone().cpu().data, 3, normalize=True)
            # ndvi = (image[:3,3:4] - image[:3,0:1]) / (image[:3,3:4] + image[:3,0:1]) 
            writer.add_image('img/1Image', grid_image1, global_step)
            writer.add_image('img/2Pred', grid_image2, global_step)
            # writer.add_image('img/3DSM', grid_image4, global_step)
            writer.add_image('img/3GT', grid_image3, global_step)
            # writer.add_image('img/5NIR', grid_image5, global_step)
            writer.add_image('img/4zero', torch.zeros([3,10,10]), global_step)