import torch.nn as nn
import torch
from torch.nn import functional as F
from torch import tensor
import gin

def get_dice_coefficient(input, target, weight=None, smooth=1.0, device='cpu'):
    # input = F.softmax(input, dim=1) # If activation is ReLU
    input = input.contiguous() # [bs, 4, 240, 240]
    target = target.contiguous() #[bs, 240, 240] --- 1 for class of interest and 0 otherwise
    intersection = (input * target).sum(dim=2).sum(dim=2) # [bs, 4]
    dice_coeff = ((2.0 * intersection + smooth) / ((input**2 + target**2).sum(dim=2).sum(dim=2) + smooth)) # [bs, 4]
    weight = weight.to(device)
    return dice_coeff * weight


def get_dice_coefficient_imp2(input, target, weight=None, smooth=1.0, device='cpu'):
    dims = tuple(range(2, target.ndimension()))
    intersection = torch.sum(input * target, dims)
    cardinality = torch.sum(input ** 2 + target ** 2, dims)
    dice_coeff = ((2. * intersection + smooth) / (cardinality + smooth))
    weight = weight.to(device)
    return dice_coeff * weight

@gin.configurable
class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0, device='cpu'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if weight == None:
            self.weight = torch.tensor([1, 1, 1, 1])
        else:
            self.weight = weight

    def forward(self, input, target):
        dice_coeff = get_dice_coefficient(input, target, self.weight, self.smooth, self.device)
        loss = 1 - (dice_coeff)
        return loss.mean()
