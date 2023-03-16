import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class SegmentationLosses(object):
    def __init__(self, q=0.1, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        # self.weight = torch.from_numpy(weight)
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.q = q

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'gce':
            print('q : ', self.q)
            return self.GeneralizedCrossEntropyLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def GeneralizedCrossEntropyLoss(self, logit, target):
        q = self.q
        n, c, h, w = logit.size()
        temp = logit[:, 0, :, :]
        logits = temp[target != self.ignore_index].unsqueeze(1)
        # print(logits.shape)
        for i in range(1, c):
            temp = logit[:, i, :, :]
            logits = torch.cat([logits, temp[target != self.ignore_index].unsqueeze(1)], dim=1)

        targets = target[target != self.ignore_index].unsqueeze(1).long()
        # print(indexes.shape)
        logits = nn.functional.softmax(logits, dim=1)
        # print(logits.shape)
        # print(targets.shape)
        Fj = torch.gather(logits, 1, targets)
        # print(Fj.shape)
        if self.weight is not None:
            if self.cuda:
                self.weight = self.weight.cuda()
            loss = torch.mean(((1 - (Fj + 1e-8) ** q) / q) * self.weight[targets])
        else:
            loss = torch.mean(((1 - (Fj + 1e-8) ** q) / q))

        # if self.batch_average:
        #     loss /= n

        return loss
        
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=255, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 2, 2).cuda()
    b = torch.rand(1, 3, 2, 2).cuda()
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print( DiceLoss()(a, b).item() )




