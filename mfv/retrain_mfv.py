import os
import pdb
import warnings
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
from modeling.sync_batchnorm.replicate import patch_replication_callback
from tqdm import tqdm
import dataloaders
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler

import build_mfv

import argparse
from mypath import Path

import warnings
warnings.filterwarnings("ignore")

def obtain_retrain_mfvnet_args():
    parser = argparse.ArgumentParser(description="MFVNet ReTraining")
    parser.add_argument('--gpu', type=str, default='0', help='test time gpu device id')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')

    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    
    parser.add_argument('--dataset', type=str, default='wfv', help='pascal or cityscapes')
    parser.add_argument('--checkname', default='Test', type=str)
    
    args = parser.parse_args()
    return args

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速
    print('=> random seed fixed to', seed)

def load_ssm_model(model, s1_path, s2_path, s3_path, s4_path=None):
    state_dict = model.state_dict()
    model_dict = {}
    ckpt_s1 = torch.load(s1_path)
    pretrain_dict = ckpt_s1['state_dict']

    for k, v in pretrain_dict.items():
        if 'module.s1_encoder.'+k in state_dict:
            name = 'module.s1_encoder.'+k  # remove 'module.' of dataparallel
            model_dict[name] = v
            
    if 'hr' in s1_path:
        model_dict['module.s1_encoder.last_conv1.weight'] = pretrain_dict['last_layer.0.weight']
        model_dict['module.s1_encoder.last_conv1.bias'] = pretrain_dict['last_layer.0.bias']
        model_dict['module.s1_encoder.last_bn.weight'] = pretrain_dict['last_layer.1.weight']
        model_dict['module.s1_encoder.last_bn.bias'] = pretrain_dict['last_layer.1.bias']
        model_dict['module.s1_encoder.last_bn.running_mean'] = pretrain_dict['last_layer.1.running_mean']
        model_dict['module.s1_encoder.last_bn.running_var'] = pretrain_dict['last_layer.1.running_var']
        model_dict['module.s1_encoder.last_bn.num_batches_tracked'] = pretrain_dict['last_layer.1.num_batches_tracked']
        model_dict['module.s1_encoder.last_conv2.weight'] = pretrain_dict['last_layer.3.weight']
        model_dict['module.s1_encoder.last_conv2.bias'] = pretrain_dict['last_layer.3.bias']

    ckpt_s2 = torch.load(s2_path)
    pretrain_dict = ckpt_s2['state_dict']
    for k, v in pretrain_dict.items():
        if 'module.s2_encoder.'+k in state_dict:
            name = 'module.s2_encoder.'+k  # remove 'module.' of dataparallel
            model_dict[name] = v
            
    if 'hr' in s2_path:
        model_dict['module.s2_encoder.last_conv1.weight'] = pretrain_dict['last_layer.0.weight']
        model_dict['module.s2_encoder.last_conv1.bias'] = pretrain_dict['last_layer.0.bias']
        model_dict['module.s2_encoder.last_bn.weight'] = pretrain_dict['last_layer.1.weight']
        model_dict['module.s2_encoder.last_bn.bias'] = pretrain_dict['last_layer.1.bias']
        model_dict['module.s2_encoder.last_bn.running_mean'] = pretrain_dict['last_layer.1.running_mean']
        model_dict['module.s2_encoder.last_bn.running_var'] = pretrain_dict['last_layer.1.running_var']
        model_dict['module.s2_encoder.last_bn.num_batches_tracked'] = pretrain_dict['last_layer.1.num_batches_tracked']
        model_dict['module.s2_encoder.last_conv2.weight'] = pretrain_dict['last_layer.3.weight']
        model_dict['module.s2_encoder.last_conv2.bias'] = pretrain_dict['last_layer.3.bias']

    ckpt_s3 = torch.load(s3_path)
    pretrain_dict = ckpt_s3['state_dict']
    for k, v in pretrain_dict.items():
        if 'module.s3_encoder.'+k in state_dict:
            name = 'module.s3_encoder.'+k  # remove 'module.' of dataparallel
            model_dict[name] = v
            
    if 'hr' in s3_path:
        model_dict['module.s3_encoder.last_conv1.weight'] = pretrain_dict['last_layer.0.weight']
        model_dict['module.s3_encoder.last_conv1.bias'] = pretrain_dict['last_layer.0.bias']
        model_dict['module.s3_encoder.last_bn.weight'] = pretrain_dict['last_layer.1.weight']
        model_dict['module.s3_encoder.last_bn.bias'] = pretrain_dict['last_layer.1.bias']
        model_dict['module.s3_encoder.last_bn.running_mean'] = pretrain_dict['last_layer.1.running_mean']
        model_dict['module.s3_encoder.last_bn.running_var'] = pretrain_dict['last_layer.1.running_var']
        model_dict['module.s3_encoder.last_bn.num_batches_tracked'] = pretrain_dict['last_layer.1.num_batches_tracked']
        model_dict['module.s3_encoder.last_conv2.weight'] = pretrain_dict['last_layer.3.weight']
        model_dict['module.s3_encoder.last_conv2.bias'] = pretrain_dict['last_layer.3.bias']

    if not s4_path == None:
        raise NotImplementedError
    
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    
    return model

def main():
    args = obtain_retrain_mfvnet_args()
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    fix_seed(4396)
    saver = Saver(args)
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()
    
    args.gpu = [int(s) for s in args.gpu.split(',')]
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': False}
    dataset_loader, val_loader, test_loader, num_classes = dataloaders.make_data_loader(args, num_scales=3, **kwargs)
    args.num_classes = num_classes

    model = build_mfv.Build_endecoder(dataset = args.dataset)
    weight = None
    criterion = SegmentationLosses(weight=weight, cuda=True).build_loss(mode=args.loss_type)

    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(dataset_loader))
    start_epoch = args.start_epoch
    evaluator = Evaluator(args.num_classes)
    best_pred = 0.0
    sec_pred = 0.0
    trd_pred = 0.0
    for_pred = 0.0
    fif_pred = 0.0

    model = nn.DataParallel(model, device_ids=args.gpu)
    patch_replication_callback(model)
    model = model.cuda()
    if args.dataset == 'gid':
        model = load_ssm_model(model, s1_path = '../ssm_models/gid_s1_psp.pth.tar', 
                                      s2_path = '../ssm_models/gid_s2_u.pth.tar',
                                      s3_path = '../ssm_models/gid_s3_hr.pth.tar')
    elif args.dataset == 'potsdam':
        model = load_ssm_model(model, s1_path = '../ssm_models/potsdam_s1_psp.pth.tar', 
                                      s2_path = '../ssm_models/potsdam_s2_psp.pth.tar',
                                      s3_path = '../ssm_models/potsdam_s3_u.pth.tar')
    elif args.dataset == 'wfv':
        model = load_ssm_model(model, s1_path = '../ssm_models/wfv_s1_hr.pth.tar', 
                                      s2_path = '../ssm_models/wfv_s2_hr.pth.tar',
                                      s3_path = '../ssm_models/wfv_s3_u.pth.tar')
    else:
        raise NotImplementedError
 
    ################################ Training ################################        
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        tbar = tqdm(dataset_loader, ncols = 80)
        num_img_tr = len(dataset_loader)
        for i, sample in enumerate(tbar):
            scheduler(optimizer, i, epoch, best_pred)
            s1_img = sample['s1_img'].cuda()
            target = sample['s1_msk'].cuda()
            s2_img = sample['s2_img'].cuda()
            s3_img = sample['s3_img'].cuda()
            output = model(s1_img, s2_img, s3_img)
            loss = criterion(output, target)

            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, Loss: %.3f]' % (epoch, train_loss))

        model.eval()
        evaluator.reset()
        tbar2 = tqdm(val_loader, ncols = 80, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar2):
            s1_img = sample['s1_img'].cuda()
            target = sample['s1_msk'].cuda()
            s2_img = sample['s2_img'].cuda()
            s3_img = sample['s3_img'].cuda()
            with torch.no_grad():
                output = model(s1_img, s2_img, s3_img)
            loss = criterion(output, target)
            test_loss += loss.item()
            tbar2.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(target, pred)

        OA = evaluator.Pixel_Accuracy()
        IoU_class, mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        Pre, Re, F1_class, F1 = evaluator.Precision_Recall_Fscore()
        Kappa = evaluator.OA_Kappa()

        if args.dataset == 'gid':
            myclass = ['buildup', 'farmland', 'forest', 'meadow', 'water']
        elif args.dataset == 'potsdam':
            myclass = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
        elif args.dataset == 'wfv':
            myclass = ['clear', 'shadow', 'cloud']
        else:
            raise NotImplementedError
        
        UA_class = dict(zip(myclass, np.array([round(a,4) for a in Pre]) ))
        F1_class = dict(zip(myclass, np.array([round(a,4) for a in F1_class]) ))
        IoU_class = dict(zip(myclass, np.array([round(a,4) for a in IoU_class]) ))
        writer.add_scalar('val/test_loss_epoch', test_loss, epoch)
        writer.add_scalar('val/OA', OA, epoch)
        writer.add_scalar('val/Kappa', Kappa, epoch)
        writer.add_scalar('val/F1', F1, epoch)
        writer.add_scalar('val/mIoU', mIoU, epoch)

        print('[Validation Epoch: %d, Loss: %.3f]' % (epoch, test_loss))
        print("Kappa: %.4f, F1: %.4f, OA: %.4f, mIoU: %.4f, FWIoU: %.4f" % (Kappa, F1, OA, mIoU, FWIoU))
        print("UA_class:{}".format(UA_class))
        print("F1_class:{}".format(F1_class))
        print("IoU_class:{}".format(IoU_class))

        new_pred = mIoU
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best, filename='checkpoint_1.pth.tar')

        elif new_pred >= sec_pred:
            is_best = False
            sec_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sec_pred': sec_pred,
            }, is_best, filename='checkpoint_2.pth.tar')

        elif new_pred >= trd_pred:
            is_best = False
            trd_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'trd_pred': trd_pred,
            }, is_best, filename='checkpoint_3.pth.tar')

        elif new_pred >= for_pred:
            is_best = False
            for_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'for_pred': for_pred,
            }, is_best, filename='checkpoint_4.pth.tar')

        elif new_pred >= fif_pred:
            is_best = False
            fif_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fif_pred': fif_pred,
            }, is_best, filename='checkpoint_5.pth.tar')

    if epoch == args.epochs-1:
        is_best = False
        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
        }, is_best, filename='checkpoint_lastep.pth.tar')
    print("Best Iou %.4f" % (best_pred))
    
    ################################ Testing ################################
    model.eval()
    print("\nStart testing")
    for ii in range(6):
        if ii == 5:
            testmodel = torch.load(os.path.join( saver.experiment_dir, 'checkpoint_lastep.pth.tar'))
        else:
            testmodel = torch.load(os.path.join( saver.experiment_dir, 'checkpoint_'+str(ii+1)+'.pth.tar'))

        test_loss = 0.0
        model.module.load_state_dict(testmodel['state_dict'])
        evaluator.reset()
        tbar3 = tqdm(test_loader, desc='\r', ncols=80)
        for i, sample in enumerate(tbar3):
            s1_img = sample['s1_img'].cuda()
            target = sample['s1_msk'].cuda()
            s2_img = sample['s2_img'].cuda()
            s3_img = sample['s3_img'].cuda()
            with torch.no_grad():
                output = model(s1_img, s2_img, s3_img)
            loss = criterion(output, target)
            test_loss += loss.item()
            tbar3.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(target, pred)

        OA = evaluator.Pixel_Accuracy()
        IoU_class, mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        Pre, Re, F1_class, F1 = evaluator.Precision_Recall_Fscore()
        Kappa = evaluator.OA_Kappa()

        if args.dataset == 'gid':
            myclass = ['buildup', 'farmland', 'forest', 'meadow', 'water']
        elif args.dataset == 'potsdam':
            myclass = ['clutter', 'imprev', 'car', 'tree', 'low', 'build']
        elif args.dataset == 'wfv':
            myclass = ['clear', 'shadow', 'cloud']
        else:
            raise NotImplementedError
        
        F1_class = dict(zip(myclass, np.array([round(a,4) for a in F1_class]) ))
        UA_class = dict(zip(myclass, np.array([round(a,4) for a in Pre]) ))
        IoU_class = dict(zip(myclass, np.array([round(a,4) for a in IoU_class]) ))
        print("Kappa: %.4f, F1: %.4f, OA: %.4f, mIoU: %.4f, FWIoU: %.4f" % (Kappa, F1, OA, mIoU, FWIoU))
        print("IoU_class:{}".format(IoU_class))
        fname = os.path.join(saver.experiment_dir,'result_test.txt')
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

if __name__ == "__main__":
    main()
