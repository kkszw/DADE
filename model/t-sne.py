import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import os
import argparse
from model.singleModel import singleTrain
from utils.dataLoader import dataLoader, weiboDataLoader, enDataLoader
import torch
import tqdm
import torch.nn.functional as F
from transformers import BertModel
from model.singleModel import singleMultiDomain
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端[1,4](@ref)
import matplotlib.pyplot as plt


def visualize_tsne_binary(features, labels, domain):
    """
    为二分类（真假新闻）创建t-SNE可视化

    参数:
    features: 高维特征矩阵 (n_samples, n_features)
    labels: 二进制标签 (0表示假新闻，1表示真新闻，或者反之)
    title: 图表标题
    """
    cate = {"科技": "Sci.",
            "军事": "Mil.",
            "教育考试": "Edu.",
            "灾难事故": "Dis/Int",
            "政治": "Pol.",
            "医药健康": "Hlth.",
            "财经商业": "Fin.",
            "文体娱乐": "Ent.",
            "社会生活": "Soc."}
    title = cate[domain],
    # 数据标准化: t-SNE对特征的尺度敏感，标准化很重要[1](@ref)
    if torch.is_tensor(features):
        if features.is_cuda:
            features = features.cpu()  # 将张量从GPU移动到CPU[6,8](@ref)
        features = features.numpy()  # 将PyTorch Tensor转换为NumPy数组[6,7](@ref)

    if torch.is_tensor(labels):
        if labels.is_cuda:
            labels = labels.cpu()
        labels = labels.numpy()
    # 确保特征数据是二维的[2,7](@ref)
    if features.ndim != 2:
        raise ValueError(f"特征数据应该是二维的，但当前维度是 {features.ndim}。请检查数据形状。")

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    # 创建t-SNE模型
    tsne = TSNE(n_components=2,
                perplexity=30,  # 困惑度，通常介于5-50，对结果影响较大[1,6](@ref)
                learning_rate=200,  # 学习率，通常在100-1000之间调整[1,7](@ref)
                n_iter=1000,  # 迭代次数，确保收敛[1](@ref)
                random_state=42)  # 随机种子，确保结果可重现

    # 执行降维并转换数据
    embeddings = tsne.fit_transform(features_std)

    # 创建图形
    plt.figure(figsize=(5, 5))

    # 为真假两类数据选择对比鲜明的颜色
    colors = ['#b99cd4', '#ffbcbb']  # 假新闻，真新闻
    label_names = ['Fake', 'True']
    total_points = 0
    # 绘制散点图
    for i, label in enumerate([0, 1]):
        mask = labels == label
        class_points = np.sum(mask)  # 计算当前类别的点数[6](@ref)
        total_points += class_points
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    c=colors[i],
                    label=label_names[i],
                    s=10,  # 点的大小
                    linewidth=0)  # 边缘线宽

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])  # 隐藏 x 轴刻度及标签
    plt.yticks([])  # 隐藏 y 轴刻度及标签
    plt.legend()
    plt.grid(False)

    # 只保存文件，不尝试显示
    plt.savefig(domain + '.png', dpi=300, bbox_inches='tight')
    # plt.savefig('tsne_binary.pdf', bbox_inches='tight')
    plt.close()  # 关闭图形释放内存
    print("is over")
    print(f"可视化完成！共绘制了 {total_points} 个数据点。")


def clip_data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'clip_text': batch[4].cuda(),
        'multi_category': batch[5].cuda()
    }
    return batch_data


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='singleModel')
parser.add_argument('--dataset', default='weibo')  # weibo21 %% weibo
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197)  # raw is 197
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--bert_vocab_file', default='../downloadModel/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
parser.add_argument('--root_path', default='../data/')
parser.add_argument('--bert', default='../downloadModel/chinese_roberta_wwm_base_ext_pytorch')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='../downloadModel/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default='../param_model')
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


def tsne(dataset, goal):
    if dataset == "weibo":
        root_path = '../data/weibo/'
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

        loader = weiboDataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                                 vocab_file=config['vocab_file'], category_dict=category_dict,
                                 num_workers=config['num_workers'], dataset='weibo')
        test_loader = loader.load_data(test_path, False)
        train_loader = loader.load_data(train_path, False)
        bert = '../downloadModel/chinese_roberta_wwm_base_ext_pytorch'
        model = singleMultiDomain(768, bert, [384], 0.2)
        model = model.cuda()
        model.load_state_dict(torch.load("../param_model/singleModel/weibo/singleModelweibo_2.pkl"))
        model.eval()
    elif dataset == "weibo21":
        root_path = '../data/weibo21/'
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

        loader = dataLoader(max_len=config['max_len'], batch_size=config['batchSize'],
                            vocab_file=config['vocab_file'], category_dict=category_dict,
                            num_workers=config['num_workers'], dataset='weibo')
        test_loader = loader.load_data(test_path, False)
        train_loader = loader.load_data(train_path, False)
        bert = '../downloadModel/chinese_roberta_wwm_base_ext_pytorch'
        model = singleMultiDomain(768, bert, [384], 0.2)
        model = model.cuda()
        model.load_state_dict(torch.load("../param_model/singleModel/weibo21/singleModelweibo21_1.pkl"))
        model.eval()

    if goal == "goal":
        pred = []
        all_features = []
        all_labels = []
        for step_n, batch in enumerate(tqdm.tqdm(train_loader)):
            with torch.no_grad():
                batch_data = clip_data2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred, _, _, _, _, feature, _ = model(**batch_data)
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())

                all_features.append(feature.cpu())
                all_labels.append(batch_label.cpu())

        all_features_tensor = torch.cat(all_features, dim=0)  # 形状变为 [总样本数, 512]
        all_labels_tensor = torch.cat(all_labels, dim=0)  # 形状变为 [总样本数]

        print(f"合并后特征矩阵形状: {all_features_tensor.shape}")
        print(f"合并后标签形状: {all_labels_tensor.shape}")

        # 修复3: 确保数据是二维的 (样本数 × 特征维数)
        if all_features_tensor.dim() > 2:
            all_features_tensor = all_features_tensor.view(all_features_tensor.size(0), -1)

        # 调用可视化函数
        visualize_tsne_binary(all_features_tensor, all_labels_tensor, "weibo_tsne")
    elif goal == "local":
        for i in range(0, 9):
            pred = []
            all_features = []
            all_labels = []
            all_domains = []
            domain = i
            domain_name = next((k for k, v in category_dict.items() if v == domain), None)
            for step_n, batch in enumerate(tqdm.tqdm(train_loader)):
                with torch.no_grad():
                    batch_data = clip_data2gpu(batch)
                    batch_label = batch_data['label']
                    batch_category = batch_data['category']
                    batch_label_pred, _, _, _, _, feature, _ = model(**batch_data)
                    pred.extend(batch_label_pred.detach().cpu().numpy().tolist())

                    all_features.append(feature.cpu())
                    all_labels.append(batch_label.cpu())
                    all_domains.append(batch_category.cpu())

            all_domains_tensor = torch.cat(all_domains, dim=0)
            all_features_tensor = torch.cat(all_features, dim=0)  # 形状变为 [总样本数, 512]
            all_labels_tensor = torch.cat(all_labels, dim=0)  # 形状变为 [总样本数]

            domain_mask = all_domains_tensor == domain
            all_features_tensor = all_features_tensor[domain_mask]
            all_labels_tensor = all_labels_tensor[domain_mask]

            print(f"合并后特征矩阵形状: {all_features_tensor.shape}")
            print(f"合并后标签形状: {all_labels_tensor.shape}")
            print(f"domain:{domain_name}")

            # 修复3: 确保数据是二维的 (样本数 × 特征维数)
            if all_features_tensor.dim() > 2:
                all_features_tensor = all_features_tensor.view(all_features_tensor.size(0), -1)

            # 调用可视化函数
            visualize_tsne_binary(all_features_tensor, all_labels_tensor, domain_name)


if __name__ == "__main__":
    dataset = "weibo21"
    goal = "local"
    tsne(dataset, goal)
