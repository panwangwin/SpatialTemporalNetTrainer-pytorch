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


class DataLoader():
    def __init__(self,args):
        data_dir=args['data_dir']
        self.data=pd.read_hdf(data_dir)
        adj_mx_dir=args['adj_mx_dir']
        with open(adj_mx_dir,'rb') as f:
            self.adj_mx=pickle.load(f)[2]
        self.train_ratio=args['train_ratio']
        self.test_ratio=args['test_ratio']
        self.val_ratio=args['val_ratio']
        self.seq_len=args['seq_len']
        self.horizon=args['horizon']
        assert (self.train_ratio+self.val_ratio+self.test_ratio==1)

    def data_process(self,add_time_in_day=True):
    if add_time_in_day:

    def generate_graph_seq2seq_io_data(self,
            df, add_time_in_day=True, add_day_in_week=False, scaler=None
    ):
        """
        Generate samples from
        :param df:
        :param x_offsets:
        :param y_offsets:
        :param add_time_in_day:
        :param add_day_in_week:
        :param scaler:
        :return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
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
        pass

    def get(self):
        def iterator():
            while 1<1:
                yield 1
        return iterator()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

    def forward(self,args):
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
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        args=yaml.load(f)
    main(args)