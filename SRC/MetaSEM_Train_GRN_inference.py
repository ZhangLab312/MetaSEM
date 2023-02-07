from SRC.MetaSEM_tool import *
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import average_precision_score,accuracy_score,f1_score,roc_curve,auc
#from EXP_Model import InferenceNet,Model,GenerativeNet
from SRC.MetaSEM_Model import *
import copy
Tensor = torch.cuda.FloatTensor
cell_name = 'mDC'
cell_names = ['mHSC-L','mHSC-GM','HepG2','hESC','mESC','mDC','mHSC-E']



random.seed(3258769933)
np.random.seed(3258769933)
torch.manual_seed(3258769933)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3258769933)

def get_AUPR(tsv_path,net_path):
    output = pd.read_csv(tsv_path, sep='\t')
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight', ascending=False)
    label = pd.read_csv(net_path,header = 0)
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1']) | set(label['Gene2'])
    output = output[output['TF'].apply(lambda x: x in TFs)]
    output = output[output['Target'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1'] + label['Gene2'])
    res_d = {}
    l = []
    p = []
    for item in (output.to_dict('records')):
        res_d[item['TF'] + item['Target']] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
        for item2 in set(label['Gene1']) | set(label['Gene2']):
            if item + item2 in label_set:
                l.append(1)
            else:
                l.append(0)
            if item + item2 in res_d:
                p.append(res_d[item + item2])
            else:
                p.append(-1)
    #u = average_precision_score(l, p) / np.mean(l)
    AUPR = average_precision_score(l, p)
    FPR,TPR,thresholds = roc_curve(l,p)
    AUROC = auc(FPR,TPR)
    return AUPR,FPR,AUROC

def extractEdgesFromMatrix(m, geneNames,TFmask):
    geneNames = np.array(geneNames)
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat*TFmask
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df

def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A= abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))
def eva_acc(A,Eva):
    temp = np.array(A)
    mean = np.median(A)
    temp = temp.reshape((-1,1))
    y = Eva.reshape((-1,1))
    acc = 0
    for i in range(temp.shape[0]):
        if temp[i] >= mean:
            temp[i] = 1
            if temp[i] == y[i]:
                acc = acc + 1
        else:
            temp[i] = 0
            if temp[i] == y[i]:
                acc = acc + 1
    print(f1_score(temp,y))
    return accuracy_score(temp,y)


class Train_inference:
    def __init__(self,opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def initalize_A(self,data):
        num_genes = data.shape[1]
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + np.random.randn(num_genes * num_genes).reshape(
            [num_genes, num_genes]) * 0.0005
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    def data_prepare(self, input_path, net_path):
        Ground_Truth = pd.read_csv(net_path, header=0)
        data = sc.read(input_path)
        gene_name = list(data.var_names)
        data_values = data.X
        Dropout_Mask = (data_values != 0).astype(float)
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -10)
        data_values = np.minimum(data_values, 10)
        med = np.median(data_values)
        sum_io = np.sum(data_values, 1)
        for idx in range(data_values.shape[0]):
            for jdx in range(data_values.shape[1]):
                data_values[idx, jdx] = np.exp(med * (data_values[idx, jdx] / sum_io[idx]))
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        if self.opt.is_label:
            for i in range(Ground_Truth.shape[0]):
                truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        else:
            truth_df = pd.DataFrame(np.ones([num_genes, num_genes]), index=data.columns, columns=data.columns)
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))
        y0 = torch.ones(size=[self.opt.net_size])
        for idx in truth_edges:
            if random.randint(1,100)>99:
                y0[idx[0]] = y0[idx[0]] + 1
                y0[idx[1]] = y0[idx[1]] + 1
        truth_matrix = y0.reshape([1,num_genes]).repeat(num_nodes,1)
        pseudo_data = torch.FloatTensor(data.values)
        return feat_train, truth_matrix,pseudo_data, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, data_values

    def train_model(self,input_path,net_path):
        feat_train, truth_matrix,pseudo_data, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name,data_values  = self.data_prepare(input_path,net_path)
        for epoch in range(20):
            train_data = TensorDataset(feat_train, truth_matrix, pseudo_data)
            dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False)
            adj_A_init = self.initalize_A(data)
            n_gene = adj_A_init.shape[0]
            adj_A_t = self._one_minus_A_t((torch.tensor(adj_A_init)).cuda())
            y0 = torch.zeros(size=adj_A_t.shape)
            for idx in truth_edges:
                y0[idx] = 1
            main_net = Inference(self.opt.net_size, 64, self.opt.net_size, 3, 3).cuda()
            meta_net = MetaGRNInference(3, adj_A_t.cpu(), y0,opt=self.opt).cuda()
            main_net_optimizer = torch.optim.Adam(main_net.parameters(), lr=self.opt.lr)
            meta_net_optimizer = torch.optim.Adam(meta_net.parameters(), lr=self.opt.lr_meta,weight_decay=0.00001)
            main_net_scheduler = torch.optim.lr_scheduler.StepLR(main_net_optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)
            meta_net_scheduler = torch.optim.lr_scheduler.StepLR(main_net_optimizer, step_size=self.opt.lr_step_size_meta, gamma=self.opt.gamma_meta)
            # train model
            meta_net.train()
            main_net.train()

            for i, batch in enumerate(dataloader):
                inputs_ori, label_true, pseudo_ori = batch
                inputs = inputs_ori.cuda()
                pseudo = pseudo_ori.cuda()
                label_true = label_true.cuda()
                meta_net_optimizer.zero_grad()
                meta_loss_unrolled_backward(main_net, main_net_optimizer, meta_net, pseudo, inputs, inputs, label_true,
                                            lr_main=0.01)

                meta_net_optimizer.step()
                y_pseudo = meta_net(inputs).detach()
                main_net_optimizer.zero_grad()
                output = main_net(inputs)
                if i == 0 :
                    pseudo_temp = (output.detach().cpu() * inputs_ori) * 0.5 + pseudo_ori
                else:
                    pseudo_temp = torch.cat((pseudo_temp, (output.detach().cpu() * pseudo_ori) * 0.5 + pseudo_ori),0)
                loss = 0.005 * main_net.soft_cross_entropy(output, y_pseudo) + 0.005 * torch.sum(main_net.Linear1.weight **2)/2
                # loss = 0.005 * main_net.soft_cross_entropy(output,y_pseudo) + 0.005 * torch.sum(main_net.Linear1.weight **2)/2
                loss.backward()
                main_net_optimizer.step()
                Ep, Epr = evaluate(meta_net.adj.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
            extractEdgesFromMatrix(meta_net.adj.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
                self.opt.tsv_path, sep='\t', index=False)
            #print('epoch', epoch, 'loss', loss.item(), 'Ep:', Ep, 'Epr:', Epr)
            AUPR, FPR, AUROC = get_AUPR(self.opt.tsv_path, net_path)
            print('epoch', epoch, 'loss', loss.item(), 'Ep:', Ep, 'Epr:', Epr, 'AUPR', AUPR, 'AUROC', AUROC)
            pseudo_data = torch.FloatTensor(pseudo_temp)
            meta_net_scheduler.step()
            main_net_scheduler.step()
            #pseudo_data = pd.DataFrame(pseudo_data).to_csv('E:/Mystudy/CODE/Singlecell/pseudo.csv')




class Train_test:
    def __init__(self,opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist',opt.input_path)

    def initalize_A(self,data):
        num_genes = data.shape[1]
        #A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            #[num_genes, num_genes])
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + np.random.randn(num_genes * num_genes).reshape(
            [num_genes, num_genes]) * 0.0005
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    def data_prepare(self, input_path, net_path):
        Ground_Truth = pd.read_csv(net_path, header=0)
        data = sc.read(input_path)
        gene_name = list(data.var_names)
        data_values = data.X
        Dropout_Mask = (data_values != 0).astype(float)
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -10)
        data_values = np.minimum(data_values, 10)
        med = np.median(data_values)
        sum_io = np.sum(data_values, 1)
        for idx in range(data_values.shape[0]):
            for jdx in range(data_values.shape[1]):
                data_values[idx, jdx] = np.exp(med * (data_values[idx, jdx] / sum_io[idx]))
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))
        y0 = torch.zeros(size=[self.opt.net_size])
        for idx in truth_edges:
            y0[idx[0]] = y0[idx[0]] + 1
            y0[idx[1]] = y0[idx[1]] + 1
        truth_matrix = y0.reshape([1,num_genes]).repeat(num_nodes,1)
        pseudo_data = torch.FloatTensor(data.values)
        return feat_train, truth_matrix,pseudo_data, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, data_values

    def train_model(self,input_path,net_path):
        feat_train, truth_matrix,pseudo_data, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name,data_values  = self.data_prepare(input_path,net_path)
        for epoch in range(20):
            train_data = TensorDataset(feat_train, truth_matrix, pseudo_data)
            dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, drop_last=False)
            adj_A_init = self.initalize_A(data)
            adj_A_t = self._one_minus_A_t((torch.tensor(adj_A_init)).cuda())
            y0 = torch.zeros(size=adj_A_t.shape)
            for idx in truth_edges:
                y0[idx] = 1
            main_net = Inference(self.opt.net_size, self.opt.batch_size, self.opt.net_size, 3, 3).cuda()
            meta_net = MetaGRNInference(3, adj_A_t.cpu(), y0,opt=self.opt).cuda()
            main_net_optimizer = torch.optim.Adam(main_net.parameters(), lr=self.opt.lr)
            meta_net_optimizer = torch.optim.Adam(meta_net.parameters(), lr=self.opt.lr_meta,weight_decay=0.00001)
            main_net_scheduler = torch.optim.lr_scheduler.StepLR(main_net_optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)
            meta_net_scheduler = torch.optim.lr_scheduler.StepLR(main_net_optimizer, step_size=self.opt.lr_step_size_meta, gamma=self.opt.gamma_meta)
            # train model
            meta_net.train()
            main_net.train()

            for i, batch in enumerate(dataloader):
                inputs_ori, label_true, pseudo_ori = batch
                inputs = inputs_ori.cuda()
                pseudo = pseudo_ori.cuda()
                label_true = label_true.cuda()
                meta_net_optimizer.zero_grad()
                meta_loss_unrolled_backward(main_net, main_net_optimizer, meta_net, pseudo, inputs, inputs, label_true,
                                            lr_main=0.01)
                meta_net_optimizer.step()
                y_pseudo = meta_net(inputs).detach()
                main_net_optimizer.zero_grad()
                output = main_net(inputs)
                if i == 0 :
                    pseudo_temp = (output.detach().cpu() * inputs_ori) * 0.5 + pseudo_ori
                else:
                    pseudo_temp = torch.cat((pseudo_temp, (output.detach().cpu() * pseudo_ori) * 0.5 + pseudo_ori),0)
                loss = 0.005 * main_net.soft_cross_entropy(output, y_pseudo) + 0.005 * torch.sum(main_net.Linear1.weight **2)/2
                loss.backward()
                main_net_optimizer.step()
                Ep, Epr = evaluate(meta_net.adj.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
            extractEdgesFromMatrix(meta_net.adj.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
                self.opt.tsv_path, sep='\t', index=False)
            AUPR, FPR, AUROC = get_AUPR(self.opt.tsv_path, self.opt.net_path)
            print('epoch', epoch, 'loss', loss.item(), 'Ep:', Ep, 'Epr:', Epr, 'AUPR', AUPR, 'AUROC', AUROC)
            pseudo_data = torch.FloatTensor(pseudo_temp)
            meta_net_scheduler.step()
            main_net_scheduler.step()


