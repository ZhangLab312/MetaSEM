import argparse
from SRC.MetaSEM_Model import *
from SRC.MetaSEM_tool import *
from SRC.MetaSEM_Train_GRN_inference import *
from SRC.MetaSEM_Train_robust import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20, help='Number of Epochs for training DeepSEM')
parser.add_argument('--task', type=str, default='GRN_inference')
parser.add_argument('--setting', type=str, default='default', help='Determine whether or not to use the default hyper-parameter')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
parser.add_argument('--data_file', type=str, help='The input scRNA-seq gene expression file.')
parser.add_argument('--net_file', type=str, default='',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')
parser.add_argument('--alpha', type=float, default=100, help='The loss coefficient for L1 norm of W, which is same as \\alpha used in our paper.')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of used for Feature Extractor.')
parser.add_argument('--lr_meta', type=float, default=1e-4, help='The learning rate of used for Adaptive Label Propagator.')
parser.add_argument('--lr_step_size', type=int, default=0.99, help='The step size of learning rate decay for Feature Extractor.')
parser.add_argument('--lr_step_size_meta', type=int, default=0.99, help='The step size of learning rate decay for Adaptive Label Propagator.')
parser.add_argument('--gamma', type=float, default=0.95, help='The decay factor of learning rate')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--save_name', type=str, default='/tmp')
parser.add_argument('--is_label',default=True,help='If there is no ground truth GRN, please set as None')
opt = parser.parse_args()
if opt.task == 'GRN_inference':
    if opt.setting == 'default':
        opt.alpha = 0.45
        opt.net_size = 1000
        opt.gamma = 0.95
        opt.gamma_meta = 0.85
        opt.lr = 1e-3
        opt.lr_step_size = 1
        opt.lr_meta = 5e-4
        opt.lr_step_size_meta = 1
        opt.batch_size = 64
        opt.cell = 'mHSC-E'
        opt.tsv_path = './Output' + opt.cell + '.tsv'
        opt.input_path = './Data\GRN_inference/' + opt.cell + '_input.csv'
        opt.net_path = './Data\GRN_inference/' + opt.cell + '_network.csv'
        model = Train_inference(opt)
        model.train_model(opt.input_path,opt.net_path)
    if opt.setting == 'test':
        opt.alpha = 0.45
        opt.net_size = 1000
        opt.gamma = 0.95
        opt.gamma_meta = 0.85
        opt.lr = 1e-3
        opt.lr_step_size = 1
        opt.lr_meta = 5e-4
        opt.lr_step_size_meta = 1
        opt.batch_size = 64
        opt.cell = 'mHSC-E'
        opt.tsv_path = './Output' + opt.cell + '.tsv'
        opt.input_path = './Data\GRN_inference/' + opt.cell + '_input.csv'
        opt.net_path = './Data\GRN_inference/' + opt.cell + '_network.csv'
        model = Train_test(opt)
        model.train_model(opt.input_path,opt.net_path)
elif opt.task == 'Robust':
    if opt.setting == 'default':
        opt.alpha = 0.7
        opt.net_size = 1000
        opt.gamma = 0.95
        opt.gamma_meta = 0.85
        opt.lr = 1e-3
        opt.lr_step_size = 1
        opt.lr_meta = 5e-4
        opt.lr_step_size_meta = 1
        opt.batch_size = 64
        opt.gene_number = '50'
        opt.sample_number = '50'
        opt.cell = 'mHSC-E'
        opt.tsv_path = './Output' + opt.cell + '.tsv'
        opt.input_path = './Data\GRN_robust/' + opt.cell + '_Gene(' + opt.sample_number + ')_'+ opt.gene_number + 'input.csv'
        opt.net_path = './Data\GRN_robust/' + opt.cell + '_' + opt.gene_number + '_network.csv'
        model = Train_roubust(opt)
        model.train_model(opt.input_path,opt.net_path)