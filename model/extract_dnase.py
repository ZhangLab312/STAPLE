import torch
from torch import nn


def conv_relu_bn(in_=1, out_=120, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_),
    )


class ExtractDnase(nn.Module):
    """
    This is the feature extraction module of Dnase signal
    """

    def __init__(self):
        super(ExtractDnase, self).__init__()
        self.branch_1 = conv_relu_bn(kernel_size=9)
        self.branch_2 = nn.Conv1d(in_channels=1, out_channels=120, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        branch_1 = self.branch_1(input)
        branch_2 = self.branch_2(input)
        branch_2 = self.relu(branch_2)
        branch_pool = self.maxpool(branch_1)
        res = torch.cat([branch_1, branch_2, branch_pool], dim=1)
        return res
