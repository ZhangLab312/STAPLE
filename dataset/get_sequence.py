import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.staple import Staple


class MydataSet(Dataset):
    def __init__(self, DNA_data, Dnase_seq, DNA_seq):
        self.data = [DNA_data, Dnase_seq, DNA_seq]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        DNA_data = self.data[0][item]
        Dnase_seq = self.data[1][item]
        DNA_seq = self.data[2][item]
        return [DNA_data, Dnase_seq, DNA_seq]


def get_sequence(dna_data, dnase_seq, dna_sequence, path_pth=None, path_save=None):
    """
    :param path_pth: the pth file's path
    :param dna_data: the dna sequence after one-hot encoding, type: Tensor
    :param dnase_seq: type: Tensor
    :param dna_sequence: the dna sequence without any process
    :return: the dna sequence, type :DataFrame
    """
    net = Staple(multy_task=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.load_state_dict(torch.load(path_pth, map_location=device))
    dataset_all = MydataSet(dna_data, dnase_seq, dna_sequence)
    batch_size = 1000
    testLoader = DataLoader(dataset_all, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    data_concat = pd.DataFrame()
    with torch.no_grad():
        for data in tqdm(testLoader):
            dna_data, dnase_signal, dna = data

            dna_data = dna_data.cuda()
            dnase_signal = dnase_signal.cuda()

            output = net(dna_data, dnase_signal)
            # remove redundant dimensions
            output = torch.squeeze(output)
            output = output.cpu().numpy()
            dna = np.array(list(dna))
            dna = np.squeeze(dna)

            data_temp = pd.DataFrame({'DNA_SEQ': dna, 'PREDICT_SCORE': output})
            # select the output which bigger than 0.5
            data_temp = data_temp.drop(data_temp[data_temp[2] <= 0.5].index)
            data_cat = pd.concat([data_cat, data_temp], ignore_index=True)

    if path_save is not None:
        data_cat.to_csv(path_save, sep=' ', index=False)
    return data_concat
