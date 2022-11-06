import numpy as np
import torch


def acc_count(output, labels):
    """
    :param output:the network's output
    :param labels: the labels of the dataset
    :return: the right count
    """
    # remove the redundant dimensions
    output = torch.squeeze(output)
    labels = torch.squeeze(labels)
    output = output.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    return (np.round(output) == labels).sum()


def to_numpy(data):
    """
    change the data to numpy
    :param data:
    :return:
    """
    data = torch.squeeze(data)
    return data.cpu().numpy()
