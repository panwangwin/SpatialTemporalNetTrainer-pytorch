# Created at 2020-02-06
# Filename:main.py
# Author:Wang Pan
# Purpose:
#

import pickle
from DataLoader import DataLoader
import utils
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import logging
import os
import time
import models

def logging_module_init(model_dir):
    logger = logging.getLogger('info')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(os.path.join(model_dir, 'info.log'))
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger



def pickle_save(filename,object):
    with open(filename,'wb') as f:
        pickle.dump(object,f)


class Process_Handler():
    def __init__(self,loader,logger,dir_args,model_args,train_args):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            logger.info('Using GPU...')
        self.dev=('cuda' if use_cuda else 'cpu')
        self.loader=loader
        self.logger=logger
        self.save_dir=dir_args['save_dir']
        self.log_dir=dir_args['log_dir']
        self.batch_size=train_args['batch_size']
        self.lr=train_args['learning_rate']
        self.loss_fn=self.set_loss(train_args['loss_fn'])
        self.model=self.set_model(model_args['model_name'])
        self.model=self.model.to(self.dev)
        if train_args['optimizer']=='SGD':
            self.optimizer=optim.SGD(self.model.parameters(),lr=self.lr)

    @staticmethod
    def set_model(model_name):
        if model_name=='FNN':
            return models.FNN()
        else:
            raise AttributeError('No Such Model!')

    @staticmethod
    def set_loss(loss_name): #todo loss and optimzer set
        if loss_name=='MSELoss':
            return nn.MSELoss()
        elif loss_name=='L1Loss':
            return nn.L1Loss()
        elif loss_name=='masked MSELoss':
            return utils.masked_mse_torch(null_val=0)
        elif loss_name=='masked MAELoss':
            return utils.masked_mae_torch(null_val=0)
        elif loss_name=='masked RMSELoss':
            return utils.masked_rmse_torch(null_val=0)
        else:
            raise AttributeError('No Such Loss!')

    def train(self):
        self.model.train()
        self.loader.set('train')
        self.logger.info('Training...')
        for i,(x,y) in enumerate(self.loader.get(self.batch_size)):
            x=torch.from_numpy(x).float()
            y=torch.from_numpy(y).float()
            x=x.to(self.dev)
            y=y.to(self.dev)
            pred=self.model(x)
            pred=self.loader.inverse_scale_data(pred)
            loss=self.loss_fn(pred,y)
            loss.backward()
            self.optimizer.step()
        self.logger.info('Training for current epoch Finished!')
        pass

    def val(self):#todo not batch feed but whole feed
        '''
        :return:
        All validate set are used
        '''
        self.logger.info('Validating')
        self.model.eval()
        self.loader.set('val')
        total_pred=[]
        total_y=[]
        for i,(x,y) in enumerate(self.loader.get(self.batch_size)):
            x=torch.from_numpy(x).float()
            x=x.to(self.dev)
            pred=self.model(x)
            total_y.append(y)
            pred=self.loader.inverse_scale_data(pred)
            total_pred.append(pred.cpu().detach().numpy())
        pred=np.concatenate(total_pred,axis=0)
        y=np.concatenate(total_y,axis=0)
        return utils.masked_mae_np(pred,y,null_val=0)

    def test(self):
        '''
        :return:
        All test set are used
        Horizon wised error check
        '''
        self.logger.info('Testing...')
        self.model.eval()
        self.loader.set('test')
        total_pred=[]
        total_y=[]
        for i,(x,y) in enumerate(self.loader.get(self.batch_size)):
            x=torch.from_numpy(x).float()
            x=x.to(self.dev)
            pred=self.model(x)
            total_y.append(y)
            pred=self.loader.inverse_scale_data(pred)
            total_pred.append(pred.cpu().detach().numpy())
        pred=np.concatenate(total_pred,axis=0)
        y=np.concatenate(total_y,axis=0)
        horizon_MAE=[]
        horizon_RMSE=[]
        for horizon in range(pred.shape[1]):
            pred_i=pred[:,horizon,:,:]
            y_i=y[:,horizon,:,:]
            horizon_MAE.append(utils.masked_mae_np(pred_i,y_i,null_val=0))
            horizon_RMSE.append(utils.masked_rmse_np(pred_i,y_i,null_val=0))
        return horizon_MAE,horizon_RMSE

    def save(self,filename):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optmz.state_dict()},
                   filename)
        return filename

    def load(self,filename):
        ckp = torch.load(filename)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optmz.load_state_dict(ckp['optimizer_state_dict'])
        return filename

def main(args):
    dir_args=args['dir']
    data_args=args['data']
    model_args=args['model']
    train_args=args['train']
    loader=DataLoader(data_args) #todo loader set and reset
    logger=logging_module_init(dir_args['log_dir'])
    handler=Process_Handler(loader,logger,dir_args,model_args,train_args)
    max_val=10
    for _ in range(train_args['epochs']):
        start_time=time.time()
        handler.train()
        val_mae=handler.val()
        end_time=time.time()
        logger.info('Epoch [{}/{}] val_mae: {:.4f}, using time {:.1f}s'.format(
            _, train_args['epochs'], val_mae, (end_time - start_time)))
        if val_mae<max_val:
            handler.save()
        if _%10==0:
            MAE,RMSE=handler.test()
            for i, each in enumerate(MAE):
                logger.info(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}".format(
                    i + 1, MAE[i], RMSE[i])
                )

    handler.load()
    MAE,RMSE=handler.test()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config_remote.yaml')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        args=yaml.load(f)
    main(args)