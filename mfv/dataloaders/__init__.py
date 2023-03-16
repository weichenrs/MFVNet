from dataloaders.datasets import pd_mfv3, gid_mfv3, wfv_mfv3, pd_mfv4, gid_mfv4, wfv_mfv4
from torch.utils.data import DataLoader

def make_data_loader(args, num_scales=3, **kwargs):
    if num_scales==3:
        if args.dataset == 'potsdam':
            train_set = pd_mfv3.pdSegmentation(args, split='train')
            val_set = pd_mfv3.pdSegmentation(args, split='val')
            test_set = pd_mfv3.pdSegmentation(args, split='test')
        elif args.dataset == 'wfv':
            train_set = wfv_mfv3.wfvSegmentation(args, split='train')
            val_set = wfv_mfv3.wfvSegmentation(args, split='val')
            test_set = wfv_mfv3.wfvSegmentation(args, split='test')
        elif args.dataset == 'gid':
            train_set = gid_mfv3.gidSegmentation(args, split='train')
            val_set = gid_mfv3.gidSegmentation(args, split='val')
            test_set = gid_mfv3.gidSegmentation(args, split='test')
        else:
            raise NotImplementedError
        
    elif num_scales==4:
        if args.dataset == 'potsdam':
            train_set = pd_mfv4.pdSegmentation(args, split='train')
            val_set = pd_mfv4.pdSegmentation(args, split='val')
            test_set = pd_mfv4.pdSegmentation(args, split='test')
        elif args.dataset == 'wfv':
            train_set = wfv_mfv4.wfvSegmentation(args, split='train')
            val_set = wfv_mfv4.wfvSegmentation(args, split='val')
            test_set = wfv_mfv4.wfvSegmentation(args, split='test')
        elif args.dataset == 'gid':
            train_set = gid_mfv4.gidSegmentation(args, split='train')
            val_set = gid_mfv4.gidSegmentation(args, split='val')
            test_set = gid_mfv4.gidSegmentation(args, split='test')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, num_class

