from torch import nn
import torch
import torch.nn.functional as F

from model.attention import Attention
from model.extract_dnase import ExtractDnase
from model.extract_inv import ExtractInv


class Staple(nn.Module):
    """
    This is the overall structure of the model
    """
    def __init__(self, multy_task=False):
        super(Staple, self).__init__()
        self.multy_task = multy_task
        self.extract_inv = ExtractInv()
        self.extract_DNaseseq = ExtractDnase()

        self.Dense_TF = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(160, 1)
        )
        self.Dense_CellType = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(320, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 6)
        )

        self.conv1 = nn.Conv1d(320, 480, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=8)
        self.maxpool2 = nn.MaxPool1d(kernel_size=8)
        self.lstm = nn.LSTM(hidden_size=160, input_size=24, bidirectional=True, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.han_model = Attention(6)

    def forward(self, input_DNA, input_Dnase):

        feature_DNA = self.extract_inv(input_DNA)
        feature_DNA = self.maxpool1(feature_DNA)

        feature_Dnase = self.extract_DNaseseq(input_Dnase)
        feature_Dnase = self.maxpool2(feature_Dnase)

        feature_DNA = self.han_model(feature_DNA)
        feature_Dnase = self.han_model(feature_Dnase)

        feature = torch.cat([feature_Dnase, feature_DNA], dim=2)
        feature_lstm, states = self.lstm(feature)

        output_TF = self.Dense_TF(feature_lstm)
        output_TF = self.sigmoid(output_TF)
        output_cell = self.Dense_CellType(feature_lstm)
        output_cell = F.softmax(output_cell, dim=-1)
        if self.multy_task:
            return output_cell, output_TF
        else:
            return output_TF
