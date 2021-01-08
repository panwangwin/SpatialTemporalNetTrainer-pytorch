# -*- coding:utf-8 -*-
# Created at 2020-04-18
# Filename:models.py
# Author:Wang Pan
# Purpose:
import torch
import torch.autograd
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, horizon, num_nodes, hidden_dim):
        super(FNN, self).__init__()
        self.iuput_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.act1 = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.hlayer1 = nn.Linear(self.seq_len * self.iuput_dim, self.hidden_dim)
        self.hlayer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.horizon * self.output_dim)

    def forward(self, x):
        '''
        :param x: (batch_size, horizon, num_nodes, input_dim)
        :return: y: (batch_size, horizon, num_nodes, output_dim)
        '''
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.num_nodes, self.seq_len * self.iuput_dim)
        x = self.hlayer1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.hlayer2(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.output_layer(x)
        x = x.reshape(-1, self.num_nodes, self.horizon, 1)
        x = x.permute(0, 2, 1, 3)
        return x


class S2SGRU(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, horizon, num_nodes, hidden_dim, layers):
        super(S2SGRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.encoder_cells = nn.ModuleList()
        self.decoder_cells = nn.ModuleList()
        self.encoder_cells.append(nn.GRUCell(self.input_dim, self.hidden_dim))
        self.decoder_cells.append(nn.GRUCell(self.output_dim, self.hidden_dim))
        for _ in range(1, self.layers):
            self.encoder_cells.append(nn.GRUCell(self.hidden_dim, self.hidden_dim))
            self.decoder_cells.append(nn.GRUCell(self.hidden_dim, self.hidden_dim))
        self.decoder_output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Encoder
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.seq_len, self.input_dim)
        batch_sz = x.shape[0]
        hidden_states = []
        for _ in range(self.layers):
            hidden_states.append(torch.zeros(batch_sz, 64).to(x.device))
        sequence_input = torch.unbind(x, axis=1)
        for each_input in sequence_input:
            current_input = each_input
            for layer in range(self.layers):
                hidden_states[layer] = self.encoder_cells[layer](current_input, hidden_states[layer])
                current_input = hidden_states[layer]

        # Decoder
        GO_symbol = torch.zeros(batch_sz, self.output_dim).to(x.device)
        outputs = []
        current_input = GO_symbol
        for _ in range(self.horizon):
            for layer in range(self.layers):
                hidden_states[layer] = self.decoder_cells[layer](current_input, hidden_states[layer])
                current_input = hidden_states[layer]
            current_input = self.decoder_output_layer(current_input)
            outputs.append(current_input)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape(-1, self.num_nodes, self.horizon, self.output_dim)
        outputs = outputs.permute(0, 2, 1, 3)
        return outputs
