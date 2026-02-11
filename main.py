import os
import argparse
from model.singleModel import singleTrain

from utils.dataLoader import dataLoader, weiboDataLoader, enDataLoader, phemeDataLoader, twitterDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='singleModel')
parser.add_argument('--dataset', default='pheme')  # weibo21 %% weibo %% en
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197)  # raw is 197
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--bert_vocab_file', default='./downloadModel/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
parser.add_argument('--root_path', default='../data/')
parser.add_argument('--bert', default='./downloadModel/roberta-base')
# ./downloadModel/chinese_roberta_wwm_base_ext_pytorch
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./downloadModel/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default='./param_model')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
vocab_file = None
emb_dim = None
if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file
if args.dataset == 'weibo':
    args.lr = 0.0001

config = {
    'use_cuda': True,
    'batchSize': args.batchSize,
    'max_len': args.max_len,
    'early_stop': args.early_stop,
    'num_workers': args.num_workers,
    'vocab_file': vocab_file,
    'emb_type': args.emb_type,
    'bert': args.bert,
    'root_path': args.root_path,
    'weight_decay': 5e-5,
    'model':
        {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
    'emb_dim': emb_dim,
    'lr': args.lr,
    'epoch': args.epoch,
    'model_name': args.model_name,
    'seed': args.seed,
    'save_param_dir': args.save_param_dir,
    'dataset': args.dataset
}

if __name__ == '__main__':
    train_path = None
    val_path = None
    test_path = None
    category_dict = {}
    if config['dataset'] == "weibo":
        root_path = './data/weibo/'
        train_path = root_path + 'train_weibo.xlsx'
        val_path = root_path + 'val_weibo.xlsx'
        test_path = root_path + 'test_weibo.xlsx'
        category_dict = {
            "经济": 0,
            "健康": 1,
            "军事": 2,
            "科学": 3,
            "政治": 4,
            "国际": 5,
            "教育": 6,
            "娱乐": 7,
            "社会": 8
        }
    elif config['dataset'] == "weibo21":
        root_path = './data/weibo21/'
        train_path = root_path + 'train_2_domain.xlsx'
        val_path = root_path + 'val_2_domain.xlsx'
        test_path = root_path + 'test_2_domain.xlsx'
        category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8
        }
    elif config['dataset'] == "en":
        root_path = './data/en/'
        train_path = root_path + 'train.pkl'
        val_path = root_path + 'val.pkl'
        test_path = root_path + 'test.pkl'
        category_dict = {
            "gossipcop": 0,
            "politifact": 1,
            "COVID": 2,
        }
    elif config['dataset'] == "twitter":
        root_path = './data/twitter/'
        train_path = root_path + 'train.csv'
        val_path = root_path + 'val.csv'
        test_path = root_path + 'test.csv'
        category_dict = {
            "twitter": 0,
        }
    elif config['dataset'] == "pheme":
        root_path = './data/pheme/'
        train_path = root_path + 'pheme_train.csv'
        val_path = root_path + 'pheme_val.csv'
        test_path = root_path + 'pheme_test.csv'
        category_dict = {
            "pheme": 0,
        }

    test_loader = None
    train_loader = None
    val_loader = None
    loader = None

    if config['dataset'] == "weibo":
        loader = weiboDataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                                 vocab_file=config['vocab_file'], category_dict=category_dict,
                                 num_workers=config['num_workers'], dataset='weibo')

    elif config['dataset'] == "weibo21":
        loader = dataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                            vocab_file=config['vocab_file'], category_dict=category_dict,
                            num_workers=config['num_workers'], dataset='weibo21')

    elif config['dataset'] == "en":
        loader = enDataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                              category_dict=category_dict, num_workers=config['num_workers'], dataset='en')

    elif config['dataset'] == "pheme":
        loader = phemeDataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                                 category_dict=category_dict, num_workers=config['num_workers'], dataset='pheme')

    elif config['dataset'] == "twitter":
        loader = twitterDataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                                   category_dict=category_dict, num_workers=config['num_workers'], dataset='twitter')

    print("load train data")
    train_loader = loader.load_data(train_path, True)
    print("load val data")
    val_loader = loader.load_data(val_path, False)
    print("load test data")
    test_loader = loader.load_data(test_path, False)

    trainer = singleTrain(emb_dim=config['emb_dim'], bert=config['bert'], lr=config['lr'], use_cuda=config['use_cuda'],
                          dropout=config['model']['mlp']['dropout'], mlp_dims=config['model']['mlp']['dims'],
                          weight_decay=config['weight_decay'], early_stop=config['early_stop'], epoch=config['epoch'],
                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                          category_dict=category_dict, dataset=config['dataset'], model_name=config['model_name'],
                          save_param_dir=os.path.join(config['save_param_dir'], config['model_name'],
                                                      config['dataset']))
    trainer.train()
