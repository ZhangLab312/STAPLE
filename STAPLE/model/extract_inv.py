import torch
from torch import nn


def conv_relu_bn(in_=4, out_=120, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_))


class ExtractInv(nn.Module):
    """
    This is the feature extraction module of DNA sequence
    """

    def __init__(self):
        super(ExtractInv, self).__init__()
        self.branch_1 = conv_relu_bn(kernel_size=15)
        self.branch_3 = conv_relu_bn(kernel_size=7)
        self.branch_5 = conv_relu_bn(kernel_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        branch_1 = self.branch_1(x)

        branch_3 = self.branch_3(x)
        branch_5 = self.branch_5(x)
        branch_pool = self.maxpool(branch_1)
        outputs = [branch_1, branch_3, branch_5, branch_pool]
        return torch.cat(outputs, dim=1)
