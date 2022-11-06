import logging
import os

import pandas as pd
import torch
import yaml

from data_loader.data_read import DataRead
from utils.path_util import get_project_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_reader = DataRead()


def test_param_load(net, pth_path):
    data_pth = torch.load(pth_path, map_location=device)
    data_model = net.state_dict()
    for k_one in data_pth.keys():
        if not k_one in ["h0", "c0", "extract_inv.h0"]:
            data_model[k_one] = data_pth[k_one]
    return data_model


def pth_param_load(net, pth_path, exclude_layer_name_prefix=None):
    """
    load the net's params
    :param net: the deeplearning model
    :param pth_path: the pth files' path
    :param exclude_layer_name_prefix: the prefix name of the module which we not need.
    :return:
    """

    data_pth = torch.load(pth_path, map_location=device)
    data_our_model = net.state_dict()

    for k_one in data_pth.keys():
        if exclude_layer_name_prefix is None:
            data_our_model[k_one] = data_pth[k_one]
        else:
            if not k_one.startswith(exclude_layer_name_prefix) and k_one != exclude_layer_name_prefix:
                data_our_model[k_one] = data_pth[k_one]
    return data_our_model


def data_split(data_path, len_train=0.8, test=False):
    assert os.path.exists(data_path) and data_path is not None
    data = pd.read_csv(data_path, sep=' ', header=None)
    if not test:
        # split the data
        data_train = data.sample(int(len(data) * len_train))
        data_test = data.sample(len(data) - len(data_train))
        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        train_dna = data_reader.dna_read(data_train[1])
        train_dnase = data_reader.dnase_read(data_train[2])
        train_labels = data_reader.tf_read(tf_labels=data_train[3])

        test_dna = data_reader.dna_read(data_test[1])
        test_dnase = data_reader.dnase_read(data_test[2])
        test_labels = data_reader.tf_read(data_test[3])

        logging.info('\ndata load successfully!')
        return {'train_dna': train_dna, 'train_dnase': train_dnase, 'train_labels': train_labels,
                'test_dna': test_dna, 'test_dnase': test_dnase, 'test_labels': test_labels}
    else:

        test_dna = data_reader.dna_read(data[1])
        test_dnase = data_reader.dnase_read(data[2])
        test_labels = data_reader.tf_read(data[3])
        return test_dna, test_dnase, test_labels


def yml_read():
    yml_file = open(get_project_path() + "/data.yml")
    data = yml_file.read()
    yml_reader = yaml.load(data, Loader=yaml.FullLoader)
    return yml_reader
