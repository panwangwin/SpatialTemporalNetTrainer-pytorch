# -*- coding:utf-8 -*-
# Created at 2021-01-11
# Filename:TorchDataloader.py
# Author:Wang Pan
# Purpose:
import os
import pickle
import numpy as np
import torch
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg
from RootPATH import root_base_dir

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph():
    graph_pkl_filename = os.path.join(root_base_dir, 'data/MetrLA/adj_mx_la.pkl')
    sensor_ids, sensor_id_to_ind, adj_mat = load_pickle(graph_pkl_filename)
    return adj_mat

def data_loader(X, Y, batch_size, shuffle=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def load_data(batch_size):
    data_dir = os.path.join(root_base_dir, 'data/MetrLA/processed')
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    print('Train', data['x_train'].shape, data['y_train'].shape)
    print('Val', data['x_val'].shape, data['y_val'].shape)
    print('Test', data['x_test'].shape, data['y_test'].shape)
    data['train_loader'] = data_loader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = data_loader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = data_loader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    #from Model.DCRNN2.lib.utils import DataLoader
    #data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    #data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    #data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler
    return data

def calculate_normalized_laplacian(adj):
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

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()