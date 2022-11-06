import logging
import os.path

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.staple import Staple
from utils.path_util import get_project_path
import shutil

from utils.data_util import data_split, pth_param_load, test_param_load
from utils.learn_util import to_numpy

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Main:
    """
    This is a class for training and testing
    """

    def __init__(self):
        self.net = Staple(multy_task=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.loss_function = nn.BCELoss()
        self.loss_function.to(device=self.device)
        self.__root_path = get_project_path()
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=0.00004)
        self.model_save = None

    def learn(self, epoch, data_path=None, test_step=2,
              model_save=None):

        assert data_path is not None
        self.model_save = model_save

        data_dict = data_split(data_path, len_train=0.8)
        if os.path.isdir(self.__root_path + "/tensorboard"):
            shutil.rmtree(self.__root_path + "/tensorboard", ignore_errors=True)
        writer = SummaryWriter(log_dir=self.__root_path + '/tensorboard')

        # record the auca
        auc_max = 0
        ProcessBar = tqdm(range(1, epoch + 1))
        ProcessBar.set_description('learning')
        for n_iter in ProcessBar:
            loss_train = self.train(data_dict['train_dna'], data_dict['train_dnase'], data_dict['train_labels'])
            writer.add_scalar('Train_Loss', loss_train, n_iter)

            if n_iter != test_step and n_iter % test_step == 0:
                auc, loss_test = self.test_byTrain(data_dict['test_dna'], data_dict['test_dnase'],
                                                   data_dict['test_labels'])
                writer.add_scalar('Test_Loss', loss_test, n_iter)
                writer.add_scalar('ROC_AUC', auc, n_iter)
                if auc_max <= auc:
                    auc_max = auc
                    if model_save is not None:
                        torch.save(self.net.state_dict(), model_save)

    def train(self, dna_sequence, dnase_signal, tf_labels):
        """
        this is a method for training, the param is the data required for network training
        the input's type: torch.Tensor
        """

        batch_size = 64
        tf_labels = torch.squeeze(tf_labels)
        dataset_all = torch.utils.data.TensorDataset(dna_sequence, dnase_signal, tf_labels)
        trainLoader = DataLoader(dataset_all, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.9)
        for data in trainLoader:
            self.net.train()
            dna_sequence, dnase_signal, tf_labels = data
            dna_sequence, dnase_signal, tf_labels = dna_sequence.to(self.device), \
                                                    dnase_signal.to(self.device), tf_labels.to(self.device)

            output = self.net(dna_sequence, dnase_signal)
            output = torch.squeeze(output)
            loss = self.loss_function(output, tf_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        scheduler.step()
        return loss.item()

    def transfer_learning(self, path_pth, exclude_layer_name_prefix=None):
        """
        This is a function for transfer learning,
        :param exclude_layer_name_prefix: the prefix name of the module
        which we not need. the fully connected layer's name is 'Dense_TF'
        :param path_pth: the pth files' path,
        and the files include the params of the net which was pretrained
        :return:
        """
        self.net.load_state_dict(
            pth_param_load(net=self.net, pth_path=path_pth, exclude_layer_name_prefix=exclude_layer_name_prefix))

    def test_byTrain(self, dna_sequence, dnase_signal, tf_labels):
        """
        this is a method for testing, the param is the data required for network testing
        the input's type: torch.Tensor
        :return:
        """
        tf_labels = torch.squeeze(tf_labels)
        auc, loss = self.__test_all(dna_sequence, dnase_signal, tf_labels)
        return auc, loss

    def test(self, data_path, model, persistence=False, performance_save='model.csv'):

        assert os.path.exists(model) and os.path.exists(data_path)
        self.net.load_state_dict(test_param_load(net=self.net, pth_path=model))
        dna, dnase, label = data_split(data_path=data_path, test=True)

        auc, loss = self.__test_all(dna, dnase, label)
        if persistence:
            pd.DataFrame({'roc_auc': [auc], 'loss': [loss.item()]}).to_csv(
                '../performance/' + performance_save, index=False)
        return auc, loss

    def __test_all(self, dna, dnase, label):
        logging.info('\ninffering ...')
        batch_size = 256
        dataset_all = torch.utils.data.TensorDataset(dna, dnase, label)
        testLoader = DataLoader(dataset_all, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.net.eval()
        # record the prediction and the labels
        prob_all = []
        label_all = []
        with torch.no_grad():
            # recorde the testing steps
            test_steps = -1
            for data in testLoader:
                test_steps += 1
                dna_sequence, dnase_signal, tf_labels = data
                dna_sequence, dnase_signal, tf_labels = dna_sequence.to(self.device), dnase_signal.to(
                    self.device), tf_labels.to(
                    self.device)
                output = self.net(dna_sequence, dnase_signal)
                output = torch.squeeze(output)
                loss = self.loss_function(output, tf_labels)
                predict = np.round(to_numpy(output))
                truth = np.round(to_numpy(tf_labels))
                prob_all.extend(predict)
                label_all.extend(truth)
        try:
            auc = roc_auc_score(prob_all, label_all)
        except ValueError:
            auc = 0
            pass
        logging.info("\nauc: {}, Loss: {}".format(auc, loss.item()))
        return auc, loss
