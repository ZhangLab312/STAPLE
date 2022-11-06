from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, Data_DNA, Data_ATAC):
        self.data = [Data_DNA, Data_ATAC]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        DNA_data = self.data[0][item]
        Data_ATAC = self.data[1][item]

        return [DNA_data, Data_ATAC]
