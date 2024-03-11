from torch import nn as nn
from torch.nn import functional as F
from pytorch_msssim import MS_SSIM, SSIM, ms_ssim, ssim
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):

    def __init__(self, loss_weight=1.0, data_range=1.0, size_average=True):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, X, Y):
        return self.loss_weight * (1 - ssim(X, Y, data_range=self.data_range, size_average=self.size_average))
