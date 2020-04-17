# -*- coding:utf-8 -*-
# Created at 2020-04-16
# Filename:utils.py
# Author:Wang Pan
# Purpose:

import torch
import numpy as np

def masked_mse_torch(null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    def loss(preds, labels, null_val=null_val):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = ~(labels==null_val)
        mask.float()
        mask = mask/torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds- labels)*(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    return loss


def masked_mae_torch(null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    def loss(preds, labels, null_val=null_val):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = ~(labels==null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    return loss

def masked_rmse_torch(null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    def loss(preds, labels, null_val=null_val):
        return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))

    return loss