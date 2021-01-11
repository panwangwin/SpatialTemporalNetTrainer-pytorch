# -*- coding:utf-8 -*-
# Created at 2020-04-13
# Filename:DataLoader.py
# Author:Wang Pan
# Purpose:
import copy
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg


class DataLoader():
    def __init__(self, args, logger):
        # initialization
        data_dir = args['data_dir']
        df = pd.read_hdf(data_dir)
        df_length = len(df)
        adj_mx_dir = args['adj_mx_dir']
        with open(adj_mx_dir, 'rb') as f:
            self.adj_mx = pickle.load(f, encoding='latin1')[2]
        train_ratio = args['train_ratio']
        test_ratio = args['test_ratio']
        val_ratio = args['val_ratio']
        train_len=int(train_ratio * df_length)
        test_len=int(test_ratio * df_length)
        val_len=int (val_ratio * df_length)
        self.logger = logger
        self.seq_len = args['seq_len']
        self.horizon = args['horizon']
        assert (train_ratio + val_ratio + test_ratio == 1)

        df_set = {'train': df[:train_len],
                  'val': df[train_len:train_len+val_len],
                  'test': df[-test_len:]}


        # Construct Graph
        graph = self.mat_to_nx(self.adj_mx)
        n = graph.number_of_nodes()
        self.num_nodes = n
        m = graph.number_of_edges()
        self.logger.info('\nGraph have %d nodes and %d links.'
                         '\nInput sequence length: %d '
                         '\nForecasting horizon: %d'
                         % (n, m, self.seq_len, self.horizon))
        self.graph = graph
        self.laplacian = self.calculate_scaled_laplacian(self.adj_mx)
        self.data = {}
        for each in df_set:
            xy = {}
            xy['x'], xy['y'] = self.construct_x_y(df_set[each])
            xy['y'] = xy['y'][..., [0]]
            self.data[each] = xy
        self.stage = None
        self.std = self.data['train']['x'][..., 0].std()
        self.mean = self.data['train']['x'][..., 0].mean()
        self.scaled_data = self.rescale_data()
        self.logger.info('\n Train set shape: x '+str(self.data['train']['x'].shape)+' y '+str(self.data['train']['y'].shape)+
                         '\n Val set shape: x '+str(self.data['val']['x'].shape)+' y '+str(self.data['val']['y'].shape)+
                         '\n Test set shape: x '+str(self.data['test']['x'].shape)+' y '+str(self.data['test']['y'].shape))


    def rescale_data(self):
        scaled_data = {}
        for each in self.data:
            temp_dict = {}
            temp_dict['x'] = copy.deepcopy(self.data[each]['x'])
            temp_dict['x'][..., 0] = copy.deepcopy((self.data[each]['x'][..., 0] - self.mean) / self.std)
            temp_dict['y'] = copy.deepcopy(self.data[each]['y'])
            temp_dict['y'][..., 0] = copy.deepcopy((self.data[each]['y'][..., 0] - self.mean) / self.std)
            scaled_data[each] = temp_dict
        return scaled_data

    def inverse_scale_data(self, data):
        data = data * self.std + self.mean
        return data

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

        x_offsets = np.arange(-self.seq_len + 1, 1)
        y_offsets = np.arange(1, self.horizon + 1)
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

    def set(self, stage):
        self.stage = stage
        return stage

    def get(self, batch_size,shuffle = True):
        '''
        :param batch_size:
        :return: shape:[batch_size,seq_len/horizon,num_nodes,input_dim]
        '''
        self.current_batch = 0
        data = self.scaled_data[self.stage]
        if shuffle:
            permute=np.random.permutation(len(data['x']))
            data['x']=data['x'][permute]
            data['y']=data['y'][permute]
        length = len(data['x'])
        batches = length / batch_size

        def iterator():
            while self.current_batch < batches:
                idx = self.current_batch * batch_size
                yield (data['x'][idx:idx + batch_size], data['y'][idx:idx + batch_size])
                self.current_batch = self.current_batch + 1

        return iterator()

    def current_stage_iter(self,batch_size):
        data = self.scaled_data[self.stage]
        length = len(data['x'])
        batches = length // batch_size
        return batches

    def calculate_normalized_laplacian(self, adj):
        """
        A must be symmetric matrix
        # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
        # D = diag(A 1)
        :param adj:
        :return:
        """
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian

    def calculate_scaled_laplacian(self, adj_mx, lambda_max=2, undirected=True):
        if undirected:
            adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        L = self.calculate_normalized_laplacian(adj_mx)
        if lambda_max is None:
            lambda_max, _ = linalg.eigsh(L, 1, which='LM')
            lambda_max = lambda_max[0]
        L = sp.csr_matrix(L)
        M, _ = L.shape
        I = sp.identity(M, format='csr', dtype=L.dtype)
        L = (2 / lambda_max * L) - I
        return L.astype(np.float32).todense()