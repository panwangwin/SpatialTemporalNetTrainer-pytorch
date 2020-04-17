# -*- coding:utf-8 -*-
# Created at 2020-04-16
# Filename:utils.py
# Author:Wang Pan
# Purpose:

import torch
import numpy as np

def masked_mse_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.not_equal(labels, null_val)
    mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.subtract(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.is_nan(loss), torch.zeros_like(loss), loss)
    return torch.reduce_mean(loss)


def masked_mae_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.not_equal(labels, null_val)
    mask = torch.cast(mask, torch.float32)
    mask /= torch.reduce_mean(mask)
    mask = torch.where(torch.is_nan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.subtract(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.is_nan(loss), torch.zeros_like(loss), loss)
    return torch.reduce_mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))