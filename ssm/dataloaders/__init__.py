from dataloaders.datasets import potsdam, gid, wfv
from torch.utils.data import DataLoader
from mypath import Path

def make_data_loader(args, root=None, **kwargs):

    if args.dataset == 'potsdam':
        train_set = potsdam.potsdamSegmentation(args, root=root, split='train')
        val_set = potsdam.potsdamSegmentation(args, root=root, split='val')
        test_set = potsdam.potsdamSegmentation(args, root=root, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'gid':
        train_set = gid.gidSegmentation(args, root=root, split='train')
        val_set = gid.gidSegmentation(args, root=root, split='val')
        test_set = gid.gidSegmentation(args, root=root, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'wfv':
        train_set = wfv.wfvSegmentation(args, root=root, split='train')
        val_set = wfv.wfvSegmentation(args, root=root, split='val')
        test_set = wfv.wfvSegmentation(args, root=root, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
