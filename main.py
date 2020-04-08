# Created at 2020-02-06
# Filename:main.py
# Author:Wang Pan
# Purpose:
#
import numpy as np
import pandas as pd
import pickle
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import argparse
import yaml
import networkx as nx
import sys
import scipy


class DataLoader():
    def __init__(self,args):
        #initialization
        data_dir=args['data_dir']
        df=pd.read_hdf(data_dir)
        df_length=len(df)
        adj_mx_dir=args['adj_mx_dir']
        with open(adj_mx_dir,'rb') as f:
            self.adj_mx=pickle.load(f)[2]
        train_ratio=args['train_ratio']
        test_ratio=args['test_ratio']
        val_ratio=args['val_ratio']
        self.seq_len=args['seq_len']
        self.horizon=args['horizon']
        assert (train_ratio+val_ratio+test_ratio==1)

        df_set={'train':df[:train_ratio*df_length],
                'val':df[train_ratio*df_length:(train_ratio+val_ratio)*df_length],
                'test':df[-test_ratio*df_length:]}

        #Construct Graph
        graph = self.mat_to_nx(self.adj_mx)
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        print('Graph have %d nodes and %d links.\n'
              'Input sequence length: %d \n'
              'Forecasting horizon: %d \n'
               % (
                  n, m, self.seq_len, self.horizon),
              file=sys.stderr)
        self.graph = graph
        self.data = {}
        for each in df_set:
            xy={}
            xy['x'],xy['y']=self.construct_x_y(each)
            self.data[each]=xy
        self.stage=None


    @staticmethod
    def mat_to_nx(adj_mat):
        g = nx.Graph()
        g.add_nodes_from(range(adj_mat.shape[0]))
        coo = scipy.sparse.coo_matrix(adj_mat)
        for u, v, _ in zip(coo.row, coo.col, coo.data):
            g.add_edge(u, v)
        assert g.number_of_nodes() == adj_mat.shape[0]
        return g

    def construct_x_y(self,
            df, add_time_in_day=True, add_day_in_week=False
    ):
        """
        Generate samples from
        :param df:
        :param add_time_in_day:
        :param add_day_in_week:
        :return:
        # x: (epoch_size, seq_len, num_nodes, input_dim)
        # y: (epoch_size, horizon, num_nodes, output_dim)
        """

        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]

        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        x_offsets=np.arange(-self.seq_len,1)
        y_offsets=np.arange(1,self.horizon)
        data = np.concatenate(data_list, axis=-1)
        # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def set(self,stage):
        self.stage=stage
        print("Now using %sing set"%(stage))
        return stage

    def get(self,batch_size):
        self.current_batch=0
        data=self.data[self.stage]
        length=len(data)
        batches=length/batch_size
        def iterator():
            while self.current_batch<batches:
                idx=self.current_batch*self.batch_size
                yield(data['x'][idx:idx+self.batch_size],data['y'][idx:idx+self.batch_size])
                self.current_batch=self.current_batch+1
        return iterator()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

    def forward(self,x):
        '''
        :param x: (batch_size,... other input dimensions)
        :return: y: (batch_size,... other output dimensions)
        '''
        pass

class Process_Handler():
    def __init__(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

def main(args):
    dir_args=args['dir']
    data_args=args['data']
    model_args=args['model']
    train_args=args['train']
    loader=DataLoader(data_args)
    handler=Process_Handler(loader,dir_args,model_args,train_args)
    max_val=1000
    val_mae=1000
    for _ in range(train_args['epochs']):
        handler.train()
        val_mae=handler.val()
        if val_mae<max_val:
            handler.save()
    handler.load()
    test_table=handler.test()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        args=yaml.load(f)
    main(args)