import numpy as np

eps = 1e-8
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + eps)
        return Acc

    def Pixel_Accuracy_Class(self):
        # recall
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1)+eps)
        # Acc = np.nanmean(Acc)
        return Acc

    def OA_Kappa(self):
        oa = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + eps)
        pc = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / \
             (self.confusion_matrix.sum() * self.confusion_matrix.sum() + eps)
        kappa = (oa-pc) / (1-pc + eps)
        return kappa

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + eps)
        return MIoU, np.nanmean(MIoU)

    def Precision_Recall_Fscore(self):
        Pre = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + eps)
        Re = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + eps)
        F1 = 2 * Pre * Re / (Pre + Re + eps)
        return Pre, Re, F1, np.nanmean(F1)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # if pre_image.ndim < 3:
        #     pre_image = np.expand_dims(pre_image,0)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
