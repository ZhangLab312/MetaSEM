import os
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import torch
from torch.nn.modules.module import Module
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn.parameter import Parameter
import math
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import torchsummary as summary
from einops.layers.torch import Rearrange



class Inference(nn.Module):
    def __init__(self,input,hidden,output,K,alpha):
        super(Inference, self).__init__()
        self.output = output
        self.K = K
        self.alpha = alpha
        self.n_class = 1000
        self.Linear1 = nn.Linear(input,hidden,bias=True)
        self.tanh = nn.Tanh()
        self.inference_get_logits = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,output)
        )


    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std

    def soft_cross_entropy(self, y_hat, y_soft, weight=None):
        if weight is None:
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.n_class
        else:
            loss = - torch.sum(torch.mul(weight, torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft))) / self.n_class
        return loss

    def get_logits(self,x):
        num_layers = len(self.inference_get_logits)
        for i,layer in enumerate(self.inference_get_logits):
            if i == num_layers - 1:
                #x = layer(x,temperature)
                x = layer(x)
            else:
                x = layer(x)
        return x
    def forward(self,x):
        x = self.tanh(self.Linear1(x))
        logits = self.inference_get_logits(x)
        return logits



class MetaGRNInference(nn.Module):
    def __init__(self,K,adj,y0,opt):
        super(MetaGRNInference, self).__init__()
        self.opt =opt
        self.K = K
        self.y0 = y0
        self.adj = nn.Parameter(Variable((adj).float(), requires_grad=True, name='adj_A'))
        self.weight = nn.Linear(1000, 128, bias=True)
        self.weight2 = nn.Linear(128, 1000, bias=False)
        self.ys = []
        y = self.y0
        for i in range(self.K):
            y = torch.matmul(self.adj,y)
            self.ys.append(y)

        self.ys = torch.stack(self.ys).transpose(0,1)

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.y0.shape[1])

            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        q = self.weight(x)

        q2 = torch.relu(q)

        alpha = self.weight2(q2)

        alpha = nn.Sigmoid()(alpha)

        b = (torch.matmul(alpha , self.adj))  + self.opt.alpha * alpha

        return b
