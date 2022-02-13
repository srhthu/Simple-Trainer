"""
Utilities of I/O, ...
"""
import json
import os
import sys
import random
import time

import numpy as np
import torch

CAIL_LABEL_NUM = {
    'labels_article': 94,
    'labels_charge': 115,
    'labels_penalty': 9
}

def read_json_line(path, n_line = None):
    """
    Read a txt file with json lines.

    Input:
        n_line: number of lines to read. Used for debug.
    """
    with open(path) as f:
        if n_line is None:
            data = [json.loads(k) for k in f]
        elif isinstance(n_line, int):
            data = [json.loads(f.readline()) for i in range(n_line)]
        else:
            raise ValueError(f'n_line should be int but get {n_line}')
    return data

def fetch_list(L, key):
    """
    Input:
        L: List of dict
        key: key of dict to fetch
    """
    return [ele[key] for ele in L]

def cuda(data:dict):
    cuda_data = {}
    for k, t in data.items():
        if isinstance(t, torch.Tensor):
            cuda_data[k] = t.cuda()
        else:
            cuda_data[k] = t
    return cuda_data

def to_cuda(data: dict):
    if not torch.cuda.is_available():
        return data
    else:
        return cuda(data)

def nest_batch(all_pred, batch_pred):
    """
    Append batch results to total numpy array.

    Args:
        all_pred (numpy.array):
            (num_sample, d1, ...)
        batch_pred (numpy.array):
            (batch_size, d1, ...)
    """
    return np.concatenate([all_pred, batch_pred], axis = 0)

def cal_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense. 
    Inputs: 
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = torch.sum(z) / torch.sum(mask)
    return torch.abs(sparsity - level)

def cal_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:     
        z -- (batch_size, sequence_length)
    """
    return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))

def compute_accuracy_with_logits(all_logits, all_labels):
    """
    Metric should start with `eval_`
    """
    all_preds = np.argmax(all_logits, axis = -1)
    num_samples = len(all_preds)
    acc = (all_preds == all_labels).sum() / num_samples
    metrics = {
        'eval_acc': acc
    }
    return metrics

def set_random_seed(seed = 2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_timestamp():
    """
    Return the time stamp. Useful for naming experiments.
    """
    time_stamp = time.strftime('%m%d_%H%M', time.localtime())
    return time_stamp