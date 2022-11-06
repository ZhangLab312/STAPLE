import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    This is the attention mechanism module
    """
    def __init__(self, gru_size):
        super(Attention, self).__init__()
        self.context = nn.Parameter(torch.Tensor(2 * gru_size, 1), requires_grad=True)
        self.dense = nn.Linear(2 * gru_size, 2 * gru_size)

    def forward(self, x, gpu=False):
        attention_outputs = torch.tanh(self.dense(x))
        weights = torch.matmul(attention_outputs, self.context)
        weights = F.softmax(weights, dim=1)

        if gpu:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))

        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        vector = torch.sum(x * weights, dim=1, keepdim=True)
        return vector
