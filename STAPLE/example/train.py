from main.train_test import Main
from utils.data_util import yml_read


cell_list = yml_read()['cell_list']

for cell in cell_list:
    tf_list = yml_read()['tf_list']
    if cell == 'A549':
        tf_list.remove('GABPA')

    for tf in tf_list:
        handler = Main()
        data_path = '../data/bulk_csv/' + cell + '_ENC_' + tf + '.data'
        model_save = '../pth/individual_learning/tl_after/' + cell + '_' + tf + '.pth'
        transfer_path = '../pth/cellType_learning/' + cell + '.pth'
        handler.transfer_learning(path_pth=transfer_path, exclude_layer_name_prefix='Dense_TF')
        handler.learn(epoch=30, data_path=data_path, test_step=1,
                      model_save=model_save)

        handler.test(data_path=data_path, model=model_save, persistence=True,
                     performance_save=cell + '_' + tf + '.csv')
