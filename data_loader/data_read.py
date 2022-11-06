import sys

import numpy as np
import torch


class DataRead:
    """
    This is the class used to read the input
    """

    def __init__(self):
        """
        This method initializes the data channel and the one hot encoding map
        """
        self.__channels = 4
        self.__gene_map = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

    def dna_read(self, dna_seq) -> torch.Tensor:
        """
        :param dna_seq: the DNA sequence, type:list or series
        :return:DNA_sequence after one hot coding
        """
        # Usually, the first dimension of data is batch_ Size and the second dimension is the BP length

        batch_size = len(dna_seq)
        bp_len = len(dna_seq[0])

        data_onehot = torch.randn(size=(batch_size, bp_len, self.__channels))
        for i in range(batch_size):
            temp = torch.zeros(size=(bp_len, self.__channels))
            if 'N' not in dna_seq[i]:
                for location, base in enumerate(dna_seq[i]):
                    temp[location] = torch.as_tensor(self.__gene_map[base])
            data_onehot[i] = temp
            self.__process_bar(i + 1, data_onehot.shape[0])
        return torch.transpose(data_onehot, 1, 2)

    def dnase_read(self, dnase) -> torch.Tensor:
        """
        to change the dnase which is string to torch.Tensor
        :param dnase: the dnase signal or the atac signal, type: Series, and the element is the string
        exsample: Series（'[1.0, 2.0]', '[5.3, 6.7]')
        :return: the tensor
        """
        result = torch.randn(size=(len(dnase), 1, 101))
        for i in dnase.index:
            temp = list()
            L = dnase[i][1:-1].split(',')
            for x in L:
                temp.append(float(x))
            result[i] = torch.tensor(data=temp)
        return result

    def tf_read(self, tf_labels) -> torch.Tensor:
        """
        to change the tf_labels which is series to Tensor
        :param tf_labels: type：Series
        :return: Tensor
        """
        return torch.tensor(np.array(tf_labels)).type(torch.FloatTensor)

    def __process_bar(self, num, total):
        """
        This is the method used to display the data reading progress bar
        :return: None
        """
        rate = float(num) / total
        rate_num = int(100 * rate)
        r = '\r[{}{}]{}%'.format('-' * rate_num, ' ' * (100 - rate_num), rate_num)
        sys.stdout.write(r)
        sys.stdout.flush()
