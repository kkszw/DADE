# author: szw
import os

import torch
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import cn_clip.clip as Cnclip
import clip


def _init_fn(worker_id):
    np.random.seed(2025)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data


def read_data(path):
    """
    根据文件后缀不同设置不同的读取方式
    """
    data = None
    if path.endswith('csv'):
        data = pd.read_csv(path, encoding="utf-8")
    elif path.endswith('pkl'):
        data = pd.read_pickle(path)
    elif path.endswith('json'):
        data = pd.read_json(path)
    elif path.endswith('xlsx'):
        data = pd.read_excel(path)
    elif path.endswith('npz'):
        data = np.load(path)
    return data


# process context
def word2input(texts, vocab_file, max_len, dataset):
    tokenizer = None
    if dataset in ['weibo', 'weibo21']:
        tokenizer = BertTokenizer(vocab_file=vocab_file)
    elif dataset in ['pheme', 'en', 'twitter']:
        vocab_file = './downloadModel/roberta-base/vocab.json'
        merges_file = "./downloadModel/roberta-base/merges.txt"
        tokenizer = RobertaTokenizer(vocab_file=vocab_file,
                                     merges_file=merges_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                          truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks


class dataLoader:
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers, dataset):
        self.data = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.dataset = dataset

    def load_data(self, path, shuffle):
        self.data = read_data(path)
        content = self.data['content'].astype('object').to_numpy()

        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(
            self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())

        categories = []
        for cat1, cat2 in zip(self.data['category'], self.data['领域']):
            if cat2 == "无领域":
                categories.append([self.category_dict[cat1]])
            else:
                categories.append([self.category_dict[cat1], self.category_dict[cat2]])

        num_domains = 9
        labels_multi_domain = torch.zeros(len(categories), num_domains)

        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1
        mul_category = labels_multi_domain

        token_ids, masks = word2input(content, self.vocab_file, self.max_len, self.dataset)
        clip_text = Cnclip.tokenize(content)
        datasets = TensorDataset(token_ids,
                                 masks,
                                 label,
                                 category,
                                 clip_text,
                                 mul_category
                                 )
        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader


class weiboDataLoader:
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers, dataset):
        self.data = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.dataset = dataset

    def load_data(self, path, shuffle):
        self.data = read_data(path)
        content = self.data['content'].astype('object').to_numpy()

        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(
            self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())

        categories = []
        for cat1 in self.data['category']:
            categories.append([self.category_dict[cat1]])

        num_domains = 9
        labels_multi_domain = torch.zeros(len(categories), num_domains)

        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1
        mul_category = labels_multi_domain

        token_ids, masks = word2input(content, self.vocab_file, self.max_len, self.dataset)
        clip_text = Cnclip.tokenize(content)

        datasets = TensorDataset(token_ids,
                                 masks,
                                 label,
                                 category,
                                 clip_text,
                                 mul_category
                                 )
        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader


class enDataLoader:
    def __init__(self, max_len, batch_size, category_dict, dataset, num_workers=2):
        self.data = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset

    def load_data(self, path, shuffle):
        self.data = df_filter(read_pkl(path), self.category_dict)
        content = self.data['content'].to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = word2input(content, "None", self.max_len, self.dataset)
        clip_text = clip.tokenize(content, truncate=True)

        categories = []
        for cat1 in self.data['category']:
            categories.append([self.category_dict[cat1]])

        num_domains = 3
        labels_multi_domain = torch.zeros(len(categories), num_domains)

        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1
        mul_category = labels_multi_domain

        dataset = TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                clip_text,
                                mul_category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader


class twitterDataLoader:
    def __init__(self, max_len, batch_size, category_dict, dataset, num_workers=2):
        self.data = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset

    def load_data(self, path, shuffle):
        self.data = read_data(path)
        content = self.data['content'].astype('object').to_numpy()

        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        if 'category' in self.data.columns:
            category = torch.tensor(self.data['category'].apply(
                lambda c: self.category_dict.get(c, 0)  # 如果类别不在字典中，默认为0
            ).to_numpy())
        else:
            # 如果没有category列，所有样本类别为0
            category = torch.zeros(len(self.data), dtype=torch.long)
        token_ids, masks = word2input(content, "None", self.max_len, self.dataset)
        clip_text = clip.tokenize(content, truncate=True)

        num_domains = 1
        categories = []
        for cat1 in self.data.get('category', ['fake_news'] * len(self.data)):
            categories.append([self.category_dict.get(cat1, 0)])

        labels_multi_domain = torch.zeros(len(categories), num_domains)
        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1
        mul_category = labels_multi_domain

        dataset = TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                clip_text,
                                mul_category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader


class phemeDataLoader:
    def __init__(self, max_len, batch_size, category_dict, dataset, num_workers=2):
        self.data = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset

    def load_data(self, path, shuffle):
        self.data = read_data(path)
        content = self.data['text'].astype('object').to_numpy()

        label = torch.tensor(self.data['target'].astype('object').astype(int).to_numpy())
        if 'category' in self.data.columns:
            category = torch.tensor(self.data['category'].apply(
                lambda c: self.category_dict.get(c, 0)  # 如果类别不在字典中，默认为0
            ).to_numpy())
        else:
            # 如果没有category列，所有样本类别为0
            category = torch.zeros(len(self.data), dtype=torch.long)
        token_ids, masks = word2input(content, "None", self.max_len, self.dataset)
        clip_text = clip.tokenize(content, truncate=True)

        num_domains = 1
        categories = []
        for cat1 in self.data.get('category', ['fake_news'] * len(self.data)):
            categories.append([self.category_dict.get(cat1, 0)])

        labels_multi_domain = torch.zeros(len(categories), num_domains)
        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1
        mul_category = labels_multi_domain

        dataset = TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                clip_text,
                                mul_category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader
