"""
Provide functions of dataset loading and torch dataset classes.

Features:
    Support small load
"""
import os
import sys
import json
from collections import OrderedDict
from pandas import read_json
import pickle
from typing import List

from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils import read_json_line


# =========== Dataset Classes ===========
class BasicDataset(Dataset):
    """
    A universal dataset class.

    Supported initialization methods:
        - by sample. List of samples (`dict` or `list/tuple`)
        - by column. pass kwargs of columns or list of columns
    
    Attributes:
        dataset: List of samples.
    """
    def __init__(self, *args, **kwargs):
        self.dataset = None
        self._sample_type = None # 0: list; 1: dict
        
        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError('No arguments found.')
        elif len(args) == 1 and len(kwargs) == 0:
            self._sample_type = self._get_sample_type(args[0][0])
            self.dataset = args[0]
        elif len(args) > 1 and len(kwargs) == 0:
            self._sample_type = 0
            self.dataset = [tuple(line) for line in zip(*args)]
        elif len(args) == 0:
            self._sample_type = 1
            num_samples = len(list(kwargs.values())[0])
            col_name = list(kwargs.keys())
            self.dataset = [{col: kwargs[col][i] for col in col_name} for i in range(num_samples)]
        else:
            raise ValueError('Arguments should be either positional or key words.')
    
    def _get_sample(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._get_sample(idx)
    
    def _get_sample_type(self, sample):
        """
        Sample type. 0: list, 1: dict.
        """        
        if isinstance(sample, list) or isinstance(sample, tuple):
            return 0
        elif isinstance(sample, dict):
            return 1
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}. Expected list, tuple and dict")


class LJP_Bert_OneLable(BasicDataset):
    """
    CAIL dataset. Return one label and rename it as `labels`.

    Data columns:
        input_ids,
        attention_mask
        token_type_ids
        labels_article
        labels_charge
        labels_penalty
    """
    def set_label_name(self, label_name):
        self.label_name = label_name
        self.output_field_map = {
            n:n for n in self.dataset[0] if not n.startswith('labels')} # map output field to original field
        self.output_field_map['labels'] = self.label_name

    def __getitem__(self, idx):
        # [warning] Should first set label name before call this function.
        _sample = self._get_sample(idx)
        return {n: _sample[m] for n,m in self.output_field_map.items()}


def read_cail_transformers(path, tokenizer: PreTrainedTokenizer, debug = False, return_tk = False):
    """
    Read one dataset split.
    """
    data = read_json_line(path, n_line = 1000 if debug else None)

    text = [ele['text'] for ele in data]
    tk_r = tokenizer(text, padding = 'max_length', truncation = True, return_tensors = 'np')

    ds = LJP_Bert_OneLable(
        input_ids = tk_r.input_ids,
        attention_mask = tk_r.attention_mask,
        token_type_ids = tk_r.token_type_ids,
        labels_article = [ele['label_article'] for ele in data],
        labels_charge = [ele['label_charge'] for ele in data],
        labels_penalty = [ele['label_penalty'] for ele in data]
    )
    if return_tk:
        return ds, tk_r
    else:
        return ds

def load_cail_dataset(folder, tokenizer, cache_path = None, overwrite = False, debug = False) -> List[LJP_Bert_OneLable]:
    """
    Load the whole dataset and manage cache.

    Input:
        folder: dataset folder or List of paths
        tokenizer: Transformers.Tokenizer, or callable.
        cache_path: the **folder** or path to save cache. If None, not use cache.
                    path should end with pkl
        overwrite: whether to overwrite the cache.
        debug: if true, only load a small number of samples.
    
    Output:
        train_ds, valid_ds, test_ds
    """
    CACHE_NAME = 'cail_transformer_ds.pkl'

    if isinstance(folder, str):
        paths = [os.path.join(folder, f'{ele}.json') for ele in ['train', 'valid', 'test']]
    elif isinstance(folder, list):
        paths = folder
    else:
        raise ValueError("folder should be a string or List")
    
    if cache_path:
        cache_path = cache_path if cache_path.endswith('.pkl') else \
                    os.path.join(cache_path, CACHE_NAME)
    
    if cache_path and os.path.exists(cache_path) and not overwrite:
        # load from cache
        print('Load data from cache')
        with open(cache_path, 'rb') as f:
            data_splits = pickle.load(f)
    else:
        # do tokenize
        data_splits = []
        for path in paths:
            if os.path.exists(path):
                ds = read_cail_transformers(path, tokenizer, debug = debug)
                data_splits.append(ds)
        # save cache
        if cache_path:
            print('Save data cache.')
            with open(cache_path, 'wb') as f:
                pickle.dump(data_splits, f)
    if len(data_splits) == 2:
        train_ds, test_ds = data_splits
        valid_ds = None
    elif len(data_splits) == 3:
        train_ds, valid_ds, test_ds = data_splits

    return train_ds, valid_ds, test_ds            

