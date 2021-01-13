# Created at 2020-02-06
# Filename:main.py
# Author:Wang Pan
# Purpose:
#

import argparse
import json
import logging
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import yaml
from dateutil import tz

import DCRNNModel
import models
import utils
from DataLoader import DataLoader


# Logging unit init
def logging_module_init(logger_dir):
    logger = logging.getLogger('info')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(logger_dir)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# Main Handler
class Process_Handler():
    def __init__(self, loader, logger, model_args, train_args):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            logger.info('Using GPU...')
        else:
            logger.info('Using CPU...')
        self.dev = ('cuda' if use_cuda else 'cpu')
        self.loader = loader
        self.logger = logger
        self.det = model_args['model_details']
        self.model = self.set_model(model_args['model_name'])
        self.model = self.model.to(self.dev)
        if 'scheduled_sampling' in model_args:
            self.schedule_sampling = True
        else:
            self.schedule_sampling = False
        self.train_args = train_args
        self.batch_size = train_args['batch_size']
        self.lr = train_args['learning_rate']
        self.loss_fn = self.set_loss(train_args['loss_fn'])
        self.set_optimizer(train_args['optimizer'])
        if train_args['lr_scheduler']:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=train_args['lr_milestones'],
                                                                     gamma=train_args['lr_decay_rate'])
        self.max_grad_norm = train_args['max_grad_norm']
        self.train_epochs = 0
        self.batch_logger_time = 20

    def set_optimizer(self, optimizer):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif optimizer == 'Adam':
            weight_decay = self.train_args['weight_decay']

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay, amsgrad=True,
                                        eps=1.0e-3)
        else:
            raise AttributeError('No such optimizer')

    def set_model(self, model_name):
        if model_name == 'FNN':
            return models.FNN(self.det['input_dim'], self.det['output_dim'], self.loader.seq_len, self.loader.horizon,
                              self.loader.num_nodes, self.det['hidden_dim'])
        elif model_name == 'S2SGRU':
            return models.S2SGRU(self.det['input_dim'], self.det['output_dim'], self.loader.seq_len,
                                 self.loader.horizon,
                                 self.loader.num_nodes, self.det['hidden_dim'], self.det['num_layers'])
        elif model_name == 'DCRNN':
            ###########
            self.graph = [self.loader.laplacian]
            self.graph = [torch.tensor(i).to(self.dev) for i in self.graph]
            return DCRNNModel.DCRNNModel(self.det['input_dim'], self.det['output_dim'], 12, 12,
                                         207, self.det['hidden_dim'], self.det['num_layers'], self.graph,
                                         self.det['order'])
        else:
            raise AttributeError('No Such Model!')

    def set_loss(self, loss_name):
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()
        elif loss_name == 'masked MSELoss':
            return utils.masked_mse_torch(null_val=0.0)
        elif loss_name == 'masked MAELoss':
            return utils.masked_mae_torch(null_val=0.0)
        elif loss_name == 'masked RMSELoss':
            return utils.masked_rmse_torch(null_val=0.0)
        else:
            raise AttributeError('No Such Loss!')

    def train(self):
        self.model.train()
        self.loader.set('train')
        self.logger.info('Training...')
        total_loss = 0
        per_iter = 375
        for i, (x, y) in enumerate(self.loader.get(self.batch_size)):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            x = x.to(self.dev)
            y = y.to(self.dev)
            self.optimizer.zero_grad()
            if self.schedule_sampling == True:
                tf = utils.DCRNN_teaching_force_calculater(self.train_epochs * per_iter + i, self.det['teaching_tao'])
                pred = self.model(x, y, tf)
            else:
                pred = self.model(x)
            loss = self.loss_fn(self.loader.inverse_scale_data(pred), self.loader.inverse_scale_data(y))
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if (i + 1) % self.batch_logger_time == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.3f}'.format(
                    self.train_epochs + 1, i + 1, per_iter, loss))
        if self.train_args['lr_scheduler']:
            self.lr_scheduler.step()
        self.logger.info('Training for current epoch Finished!')
        self.train_epochs += 1
        return total_loss / per_iter

    def val(self):
        """
        :return:
        """
        self.logger.info('Validating')
        self.model.eval()
        self.loader.set('val')
        total_pred = []
        total_y = []
        total_loss = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.loader.get(self.batch_size, shuffle=False)):
                x = torch.from_numpy(x).float()
                x = x.to(self.dev)
                if self.schedule_sampling == True:
                    y = torch.from_numpy(y).float()
                    y = y.to(self.dev)
                    pred = self.model(x, y, teaching_force=0)
                    y = y.cpu().detach().numpy()
                else:
                    pred = self.model(x)
                y = self.loader.inverse_scale_data(y)
                total_y.append(y)
                pred = self.loader.inverse_scale_data(pred)
                total_pred.append(pred.cpu().detach().numpy())
                total_loss.append(utils.masked_mae_np(pred.cpu().detach().numpy(), y, null_val=0.0))
            pred = np.concatenate(total_pred, axis=0)
            y = np.concatenate(total_y, axis=0)
        return utils.masked_mae_np(pred, y, null_val=0.0), np.mean(total_loss)

    def test(self):
        """
        :return:
        """
        self.logger.info('Testing...')
        self.model.eval()
        self.loader.set('test')
        total_pred = []
        total_y = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.loader.get(self.batch_size, shuffle=False)):
                x = torch.from_numpy(x).float()
                x = x.to(self.dev)
                if self.schedule_sampling == True:
                    y = torch.from_numpy(y).float()
                    y = y.to(self.dev)
                    pred = self.model(x, y, teaching_force=0)
                    y = y.cpu().detach().numpy()
                else:
                    pred = self.model(x)
                total_y.append(self.loader.inverse_scale_data(y))
                pred = self.loader.inverse_scale_data(pred)
                total_pred.append(pred.cpu().detach().numpy())
            pred = np.concatenate(total_pred, axis=0)
            y = np.concatenate(total_y, axis=0)
            horizon_MAE = []
            horizon_RMSE = []
            for horizon in range(pred.shape[1]):
                pred_i = pred[:, horizon, :, :]
                y_i = y[:, horizon, :, :]
                horizon_MAE.append(utils.masked_mae_np(pred_i, y_i, null_val=0.0))
                horizon_RMSE.append(utils.masked_rmse_np(pred_i, y_i, null_val=0.0))
        return horizon_MAE, horizon_RMSE

    def save(self, filename):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   filename)
        return filename

    def load(self, filename):
        ckp = torch.load(filename)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        return filename


def main(args, status):
    dir_args = args['dir']
    data_args = args['data']
    model_args = args['model']
    train_args = args['train']
    max_val = 100000

    if status == 'Train':
        model_dir = dir_args['base_dir'] + '/model_%s_%s' % (
        model_args['model_name'], str(pd.datetime.now(tz=tz.gettz('Asia/Shanghai'))))
        os.mkdir(model_dir)
        dir_args['model_dir'] = model_dir
        logger = logging_module_init(model_dir + '/info_train.log')
        logger.info('\n NOW TRAINING WITH FOLLOWING PARAMETERS:'
                    '\n %s' % (json.dumps(args, indent=4)))
        loader = DataLoader(data_args, logger)
        # loader=load_data(64)
        try:
            handler = Process_Handler(loader, logger, model_args, train_args)
            for _ in range(train_args['epochs']):
                start_time = time.time()
                train_loss = handler.train()
                logger.info('Current epoch train loss %.4f' % train_loss)
                val_mae, mean_val_mae = handler.val()
                model_file = model_dir + '/model_%s_epoch_%d_val_mae_%.4f' % (model_args['model_name'], _ + 1, val_mae)
                end_time = time.time()
                logger.info('Epoch [{}/{}] val_mae: {:.4f}, mean_val_mae: {:.4f} using time {:.1f}s'.format(
                    _ + 1, train_args['epochs'], val_mae, mean_val_mae, (end_time - start_time)))
                if val_mae < max_val:
                    best_model_file = model_dir + '/current_best_%s_epoch_%d_val_mae_%.4f' % (
                    model_args['model_name'], _ + 1, val_mae)
                    dir_args['best_model_dir'] = best_model_file
                    with open(model_dir + '/config_test.yaml', 'w') as f:
                        yaml.dump(args, f)
                    handler.save(best_model_file)
                    max_val = val_mae
                if (_ + 1) % 5 == 0:
                    MAE, RMSE = handler.test()
                    for i, each in enumerate(MAE):
                        logger.info(
                            "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}".format(
                                i + 1, MAE[i], RMSE[i])
                        )
        except:
            logger.error('\n' + traceback.format_exc())

    if status == 'Test':
        logger = logging_module_init(dir_args['model_dir'] + '/info_test.log')
        loader = DataLoader(data_args, logger)
        logger.info('\n NOW TESTING WITH MODELS TRAINING BY FOLLOWING PARAMETERS:'
                    '\n %s' % (json.dumps(args, indent=4)))
        handler = Process_Handler(loader, logger, model_args, train_args)
        best_model_file = dir_args['best_model_dir']
        handler.load(best_model_file)
        MAE, RMSE = handler.test()
        for i, each in enumerate(MAE):
            logger.info(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}".format(
                    i + 1, MAE[i], RMSE[i])
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config_remote_train_DCRNN.yaml')
    parser.add_argument('--status', default='Train')
    args = parser.parse_args()
    status = args.status
    with open(args.config, 'r') as f:
        model_args = yaml.load(f)
    main(model_args, status)
