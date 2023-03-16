import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.unet import UNet
from modeling.seg_hrnet import *
from modeling.pspnet import *

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import warnings
warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':False}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, 
                    root=Path.db_root_dir(dataset=self.args.dataset, scale=self.args.ssm_scale), **kwargs)

        weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # choose ssm model
        if self.args.ssm_model == 'deeplab':
        ## Deeplab
            model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
            # Define Optimizer
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
            #optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay, lr=args.lr)
        else:
            ## HRNet
            if self.args.ssm_model == 'hrnet':
                model = HighResolutionNet(num_classes=self.nclass)
            ## UNet
            elif self.args.ssm_model == 'unet':
                model = UNet(n_channels=4, n_classes=self.nclass)
            ## PSPNet
            elif self.args.ssm_model == 'pspnet':
                model = PSPNet(layers=101, pretrained=False, classes=self.nclass, criterion=self.criterion)
            else:
                raise NotImplementedError
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, 
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
            
        self.model, self.optimizer = model, optimizer
        # Define evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_pred2 = 0.0
        self.sec_pred = 0.0
        self.trd_pred = 0.0
        self.for_pred = 0.0
        self.fif_pred = 0.0

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def get_gid_labels(self):
        return np.array([
            [255,0,0],    #buildup
            [0,255,0],   #farmland
            [0,255,255],  #forest
            [255,255,0],  #meadow
            [0,0,255] ])  #water
        
    def get_potsdam_labels():
        return np.array([
            [255,0,0],    #clutter
            [255,255,255],  #imprevious
            [255,255,0],  #car
            [0,255,0],   #tree
            [0,255,255],  #low vegetation
            [0,0,255] ])  #building

    def decode_segmap(self, label_mask, dataset):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        if dataset == 'gid':
            n_classes = 5
            label_colours = self.get_gid_labels()
        elif dataset == 'potsdam':
            n_classes = 6
            label_colours = self.get_potsdam_labels()
        else:
            raise NotImplementedError

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader, ncols=80)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            # for PSPNet
            if self.args.ssm_model == 'pspnet':
                main_loss, aux_loss = self.model(image, target)
                loss = main_loss + 0.4 * aux_loss
            ## for the other networks
            else:
                output = self.model(image)
                loss = self.criterion(output, target)
                
            # Show 10 * 3 inference results each epoch
            if self.args.show_intermediate == True:
                if i % (num_img_tr // 2) == 0:
                    global_step = i + num_img_tr * epoch
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
            
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        OA = self.evaluator.Pixel_Accuracy()
        IoU_class, mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        Pre, Re, F1_class, F1 = self.evaluator.Precision_Recall_Fscore()
        Kappa = self.evaluator.OA_Kappa()
        if self.args.dataset == 'gid':
            myclass = ['buildup', 'farmland', 'forest', 'meadow', 'water']
        elif self.args.dataset == 'potsdam':
            myclass = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
        elif self.args.dataset == 'wfv':
            myclass = ['clear', 'shadow', 'cloud']
        else:
            raise NotImplementedError

        IoU_class = dict(zip(myclass, np.array([round(a,4) for a in IoU_class]) ))
        self.writer.add_scalar('val/test_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/OA', OA, epoch)
        self.writer.add_scalar('val/Kappa', Kappa, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/FWIoU', FWIoU, epoch)
        self.writer.add_scalars('val/IoU_class', IoU_class, epoch)

        print('[Validation Epoch: %d, Loss: %.3f]' % (epoch, test_loss))
        print("Kappa: %.4f, F1: %.4f, OA: %.4f, mIoU: %.4f, FWIoU: %.4f" % (Kappa, F1, OA, mIoU, FWIoU))
        print("IoU_class:{}".format(IoU_class))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='checkpoint_1.pth.tar')

        elif new_pred >= self.sec_pred:
            is_best = False
            self.sec_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'sec_pred': self.sec_pred,
            }, is_best, filename='checkpoint_2.pth.tar')

        elif new_pred >= self.trd_pred:
            is_best = False
            self.trd_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'trd_pred': self.trd_pred,
            }, is_best, filename='checkpoint_3.pth.tar')

        elif new_pred >= self.for_pred:
            is_best = False
            self.for_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'for_pred': self.for_pred,
            }, is_best, filename='checkpoint_4.pth.tar')

        elif new_pred >= self.fif_pred:
            is_best = False
            self.fif_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'fif_pred': self.fif_pred,
            }, is_best, filename='checkpoint_5.pth.tar')
        # print('Val pred/best:{}/{}'.format(new_pred,self.best_pred))
        print('Val pred/best:%.4f/%.4f'%(new_pred,self.best_pred))

    def test(self):
        self.model.eval()
        for ii in range(5):
            print("\nStart testing")
            testmodel = torch.load(os.path.join( self.saver.experiment_dir, 'checkpoint_'+str(ii+1)+'.pth.tar'))
            # testmodel = torch.load(os.path.join( self.saver.directory, 'model_best.pth.tar'))
            self.model.module.load_state_dict(testmodel['state_dict'])
            self.evaluator.reset()
            tbar = tqdm(self.test_loader, desc='\r', ncols=80)
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                name = sample['name']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                mask = target.squeeze().astype(np.uint8)
                self.evaluator.add_batch(mask, pred)

                if self.args.predict == True:
                    if self.args.dataset == 'gid':
                        pred_col = self.decode_segmap(pred.squeeze(), 'gid')
                        pred_col[:,:,0][mask==255] = 0
                        pred_col[:,:,1][mask==255] = 0
                        pred_col[:,:,2][mask==255] = 0
                        pred_col = Image.fromarray(np.uint8(pred_col))
                    elif self.args.dataset == 'potsdam':
                        pred_col = self.decode_segmap(pred.squeeze(), 'potsdam')
                        pred_col[:,:,0][mask==255] = 0
                        pred_col[:,:,1][mask==255] = 0
                        pred_col[:,:,2][mask==255] = 0
                        pred_col = Image.fromarray(np.uint8(pred_col))
                    elif self.args.dataset == 'wfv':
                        _tmp = pred.clone()
                        _tmp[_tmp == 2] = 255   # cloud
                        _tmp[_tmp == 1] = 128   # shadow  
                        _tmp[_tmp == 0] = 1     # clear
                        _tmp[target == 0] = 0   
                        pred_col = Image.fromarray(np.uint8(_tmp))
                        
                    pred_col.save('%s/%s_color.png' % ('pred', name[0].split('.tif')[0] ))
                
            OA = self.evaluator.Pixel_Accuracy()
            IoU_class, mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            Pre, Re, F1_class, F1 = self.evaluator.Precision_Recall_Fscore()
            Kappa = self.evaluator.OA_Kappa()
            
            if self.args.dataset == 'gid':
                myclass = ['buildup', 'farmland', 'forest', 'meadow', 'water']
            elif self.args.dataset == 'potsdam':
                myclass = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
            elif self.args.dataset == 'wfv':
                myclass = ['clear', 'shadow', 'cloud']
            else:
                raise NotImplementedError
            
            F1_class = dict(zip(myclass, np.array([round(a,4) for a in F1_class]) ))
            UA_class = dict(zip(myclass, np.array([round(a,4) for a in Pre]) ))
            IoU_class = dict(zip(myclass, np.array([round(a,4) for a in IoU_class]) ))
            print("Kappa: %.4f, F1: %.4f, OA: %.4f, mIoU: %.4f, FWIoU: %.4f" % (Kappa, F1, OA, mIoU, FWIoU))

            print("IoU_class:{}".format(IoU_class))
            fname = os.path.join(self.saver.experiment_dir,'result_test.txt')
            f = open(fname, 'a')
            f.write("Kappa: %.4f, F1: %.4f, OA: %.4f, mIoU: %.4f, FWIoU: %.4f" % (Kappa, F1, OA, mIoU, FWIoU))
            f.write('\n')
            f.write("UA_class:{}".format(UA_class))
            f.write('\n')
            f.write("F1_class:{}".format(F1_class))
            f.write('\n')
            f.write("IoU_class:{}".format(IoU_class))
            f.write('\n\n')
            f.close()

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset', type=str, default='potsdam',
                        choices=['potsdam', 'wfv', 'gid'],
                        help='dataset name (default: potsdam)')
    parser.add_argument('--ssm-scale', type=str, default='512',
                        choices=['512', '768', '1024', '1280'],
                        help='ssm scale')
    parser.add_argument('--ssm-model', type=str, default='hrnet',
                        choices=['hrnet', 'pspnet', 'unet', 'deeplab'],
                        help='ssm model name (default: hrnet)')

    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # parameters for Deeplab
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name for deeplab (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride for deeplab (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # parameters for Deeplab
    
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='MONTH_DAY_DATASET_MODEL_SCALE',
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # whether to output the predictions while testing
    parser.add_argument('--show-intermediate', action='store_true', default=False,
                        help='show intermediate training results')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='output the predictions')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.checkname is None:
        args.checkname = 'testing'
        
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()
    trainer.test()
    
if __name__ == "__main__":
    main()
