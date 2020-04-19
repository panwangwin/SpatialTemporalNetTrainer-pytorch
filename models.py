# -*- coding:utf-8 -*-
# Created at 2020-04-18
# Filename:models.py
# Author:Wang Pan
# Purpose:
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

class FNN(nn.Module): #todo find the essence of the batch
    def __init__(self):
        super(FNN,self).__init__()
        self.iuput_dim=2
        self.output_dim=1
        self.num_nodes=207
        self.layer1=nn.Linear(self.iuput_dim*self.num_nodes*12,self.num_nodes*24)
        self.layer2=nn.Linear(self.num_nodes*24,self.num_nodes*12)
    def forward(self,x):
        '''
        :param x: (batch_size,... other input dimensions)
        :return: y: (batch_size,... other output dimensions)
        '''
        batch_sz=x.shape[0]
        x=x.reshape(batch_sz,self.iuput_dim*self.num_nodes*12)
        x=self.layer1(x)
        x=torch.tanh(x)
        x=self.layer2(x)
        x=torch.tanh(x)
        x=x.reshape(batch_sz,12,207,1)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

    def forward(self, *input):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

    def forward(self, *input):
        pass

class GRUCell(nn.Module):
    def __init__(self):
        super(GRUCell,self).__init__()

    def forward(self, *input):
        pass

