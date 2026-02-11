# weibo 和 weibo21表现都好
import json
import os
import clip
import torch
import tqdm
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from cn_clip.clip import load_from_name
from model.layers import *


class DomainExpertEnhancer(nn.Module):
    def __init__(self, domain_num, num_expert, feature_dim):
        super().__init__()
        self.domain_num = domain_num
        self.num_expert = num_expert

        self.text_experts = nn.ModuleList([
            nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_expert)])
            for _ in range(domain_num)
        ])

        self.text_share_expert = nn.ModuleList([
            nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_expert * 2)])
        ])

        self.text_gate_list = nn.ModuleList([
            nn.Linear(feature_dim, num_expert * 3)  # 3 = num_expert + num_share_expert
            for _ in range(domain_num)
        ])

        self.att_mlp_text = nn.Sequential(
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )

        self.contrastive_loss = nn.CosineEmbeddingLoss()

        self.intra_att = nn.Linear(feature_dim, 1)
        self.inter_att = nn.Linear(feature_dim, 1)

    def forward(self, text_prime, labels, category):
        domain_features = []
        total_loss = 0

        for i in range(self.domain_num):
            domain_mask = (category == i)
            T_domain = torch.zeros_like(text_prime)
            T_domain[domain_mask] = text_prime[domain_mask]

            gate_weights = self.text_gate_list[i](T_domain)  # [64, 9]

            expert_output = 0
            for j in range(self.num_expert):
                expert_feat = self.text_experts[i][j](T_domain)
                expert_output += expert_feat * gate_weights[:, j].unsqueeze(1)

            share_output = 0
            for j in range(self.num_expert * 2):
                share_feat = self.text_share_expert[0][j](T_domain)
                expert_output += share_feat * gate_weights[:, self.num_expert + j].unsqueeze(1)
                share_output += share_feat * gate_weights[:, self.num_expert + j].unsqueeze(1)

            att_weights = self.att_mlp_text(expert_output)  # [64, 2]

            enhanced_feat = (att_weights[:, 0].unsqueeze(-1) * expert_output +
                             att_weights[:, 1].unsqueeze(-1) * share_output)
            # 4. 真假新闻差异增强
            if domain_mask.sum() > 0:
                pos_mask = (labels == 1) & domain_mask
                neg_mask = (labels == 0) & domain_mask
                feat_pos = enhanced_feat[pos_mask]
                feat_neg = enhanced_feat[neg_mask]

                # 多粒度约束
                if len(feat_pos) > 0 and len(feat_neg) > 0:
                    # 新闻级差异（MMD）
                    mmd_loss = -gaussian_kernel(feat_pos, feat_neg)  # 负号表示最大化差异

                    # 特征级差异（对比学习）
                    contrast_labels = torch.cat([
                        torch.ones(len(feat_pos)),
                        -torch.ones(len(feat_neg))
                    ], dim=0).to(text_prime.device)
                    contrast_loss = self.contrastive_loss(
                        torch.cat([feat_pos, feat_neg], dim=0),
                        torch.cat([feat_neg, feat_pos], dim=0),
                        contrast_labels
                    )

                    total_loss += mmd_loss + contrast_loss

            domain_features.append(enhanced_feat)

        x = torch.stack(domain_features, dim=1)
        intra_weights = F.softmax(self.intra_att(x), dim=2)  # [64,9,1]
        intra_feat = intra_weights * x  # [64,9,512]

        inter_weights = F.softmax(self.inter_att(intra_feat), dim=1)  # [64,9,1]
        expert_feature = (inter_weights * intra_feat).sum(dim=1)  # [64,512]
        return expert_feature, 0.05 * total_loss


def gaussian_kernel(pos, neg, sigma=1.0):
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
    xx = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    dist = torch.cdist(neg.unsqueeze(0), neg.unsqueeze(0)).squeeze(0)
    yy = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    dist = torch.cdist(pos.unsqueeze(0), neg.unsqueeze(0)).squeeze(0)
    xy = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    return xx.mean() + yy.mean() - 2 * xy.mean()


class IterativeAttentionalFeatureFusion(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B,512,L] -> [B,512,1]
            nn.Conv1d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.BatchNorm1d(embed_dim // reduction_ratio),
            nn.ReLU(),
            nn.Conv1d(embed_dim // reduction_ratio, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.BatchNorm1d(embed_dim // reduction_ratio),
            nn.ReLU(),
            nn.Conv1d(embed_dim // reduction_ratio, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x, y):
        g0 = self.global_branch(x.unsqueeze(-1) + y.unsqueeze(-1)).squeeze(-1)
        l0 = self.local_branch(x.unsqueeze(-1) + y.unsqueeze(-1)).squeeze(-1)
        k = torch.sigmoid(g0 + l0)
        x_tilde = x * k
        y_tilde = y * (1 - k)  # [64,512,1]
        g1 = self.global_branch(x_tilde.unsqueeze(-1) + y_tilde.unsqueeze(-1)).squeeze(-1)
        l1 = self.local_branch(x_tilde.unsqueeze(-1) + y_tilde.unsqueeze(-1)).squeeze(-1)
        k_tilde = torch.sigmoid(g1 + l1)
        return x * k_tilde + y * (1 - k_tilde)


class singleMultiDomain(torch.nn.Module):
    def __init__(self, emb_dim, bert, mlp_dims, dropout, dataset):
        super(singleMultiDomain, self).__init__()
        self.dataset = dataset
        self.bert_dim = emb_dim  # 768
        self.emb_dim = 512  # 512
        self.num_heads = 8
        self.domain_num = 9
        self.num_expert = 3
        self.head_dim = self.emb_dim // self.num_heads  # 64
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        if self.dataset in ['pheme', 'en', 'twitter']:
            self.bert = RobertaModel.from_pretrained('roberta-base', cache_dir='./downloads/')
            self.ClipModel, _ = clip.load('ViT-B/32', device="cuda")
        elif self.dataset in ['weibo', 'weibo21']:
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
            self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        self.cnn = cnn_extractor(input_size=emb_dim, feature_kernel=feature_kernel)
        self.text_attention = MaskAttention(self.emb_dim)
        self.clip_attention = TokenAttention(self.emb_dim)
        self.bertLine = nn.Linear(self.bert_dim, self.emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.multiHead_line = MLP_fusion(4096, 512, [2048], 0.1)
        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num, embedding_dim=self.emb_dim)

        self.fine_gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
        self.coarse_gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(2 * self.emb_dim, 2 * self.emb_dim),
            nn.BatchNorm1d(2 * self.emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.emb_dim, self.emb_dim)
        )
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.mask_threshold = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
            nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, 1),
            nn.Sigmoid()
        )

        self.iAFF = IterativeAttentionalFeatureFusion(self.emb_dim)

        self.DEE = DomainExpertEnhancer(9, 3, self.emb_dim)

        self.fusion = MLP_fusion(self.emb_dim * 2, self.emb_dim, [348], 0.1)
        self.final_classifier = MLP(512, mlp_dims, dropout)

    def multiHead(self, q, k, v):
        o = []
        for i in range(self.num_heads):
            query = self.q_proj(q)
            key = self.k_proj(k)
            value = self.v_proj(v)
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
            scores = F.softmax(attn_scores, dim=-1)
            theta = self.mask_threshold(torch.cat([query, key], dim=-1))
            omega = (scores >= theta).float()
            masked_attn = F.softmax(omega * attn_scores, dim=-1)
            masked_attn = self.dropout(masked_attn)
            o.append(torch.matmul(masked_attn, value))
        output = torch.cat([o[0], o[1]], dim=-1)  # (64, 4096)
        for i in range(2, self.num_heads):
            output = torch.cat([output, o[i]], dim=-1)
        output = self.multiHead_line(output)  # (64, 512)
        return output

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        label = kwargs['label']
        bert_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        bert_feature = add_gaussian_noise(bert_feature)
        clip_text = kwargs['clip_text']
        with torch.no_grad():
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            clip_feature = clip_text_feature / clip_text_feature.norm(dim=-1, keepdim=True)
            clip_feature = clip_feature.float()  # ([64, 512])

        bert_feature = self.bertLine(bert_feature)  # ([64, 197, 768]) -> ([64, 197, 512])
        text_atn_feature = self.text_attention(bert_feature, masks)  # ([64, 512])
        clip_atn_feature = self.clip_attention(clip_feature)
        # clip_atn_feature = text_atn_feature   # BERT only
        # text_atn_feature = clip_atn_feature   # CLIP only

        indexes = torch.tensor([index for index in category]).view(-1, 1).cuda()  # [64,1]
        domain_embedding = self.domain_embedder(indexes).squeeze(1)  # ([64, 512])

        text_embedding = torch.cat([domain_embedding, text_atn_feature], dim=-1)  # ([64, 1024])
        coarse_embedding = torch.cat([domain_embedding, clip_atn_feature], dim=-1)  # ([64, 1024])
        text_fine_feature = self.fine_gate(text_embedding)  # ([64, 512])
        text_coarse_feature = self.coarse_gate(coarse_embedding)  # ([64, 512])
        O_fc = self.multiHead(text_fine_feature, text_coarse_feature, text_coarse_feature)  # ([64, 512])
        O_cf = self.multiHead(text_coarse_feature, text_fine_feature, text_fine_feature)
        text_fc = text_fine_feature + O_fc * self.residual(torch.cat([text_fine_feature, O_fc], dim=-1))  # ->([64,512])
        text_cf = text_coarse_feature + O_cf * self.residual(torch.cat([text_coarse_feature, O_cf], dim=-1))

        text_o = self.projection_head(torch.cat([text_fine_feature, clip_feature], dim=-1))  # ->([64, 512])
        text = self.projection_head(torch.cat([text_fc, text_cf], dim=-1))
        # DAMA
        # text_o = text_atn_feature
        # text = clip_atn_feature

        text_prime = self.iAFF(text_o, text)  # ->([64, 512])
        text_prime1 = text_prime
        # GRFE
        # text_prime = torch.add(text_o, text)
        Loss = 0
        domain_features, Loss = self.DEE(text_prime, label, category)  # MEDA
        text_prime = self.fusion(torch.cat([text_prime, domain_features], dim=-1))
        # MEDA
        # text_prime = text_prime1

        final_label = torch.sigmoid(self.final_classifier(text_prime).squeeze())
        return final_label, text_atn_feature, clip_atn_feature, text_fc, text_cf, text_prime, Loss


class singleTrain:
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 early_stop,
                 epoch,
                 dataset,
                 model_name,
                 loss_weight=None,
                 ):

        self.model = None
        self.batchSize = 64
        if loss_weight is None:
            loss_weight = [1, 0.006, 0.009, 5e-5]
        self.lr = lr
        self.dataset = dataset
        self.model_name = model_name
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoch = epoch
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir)
        self.save_param_dir = save_param_dir

    def train(self):
        i = 0
        while True:
            log_path = os.path.join(self.save_param_dir, f"log{i}")
            if not os.path.exists(log_path):
                with open(log_path, 'w'):
                    pass
                print(f"创建文件: {log_path}")
                break
            i += 1
        self.model = singleMultiDomain(self.emb_dim, self.bert, self.mlp_dims, self.dropout, self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoch):
            avg_loss = Averager()
            self.model.train()
            for step_n, batch in enumerate(tqdm.tqdm(self.train_loader)):
                batch_data = clip_data2gpu(batch)
                label = batch_data['label']
                final_label, text_atn_feature, clip_feature, text_fc, text_cf, _, loss0 = self.model(**batch_data)
                loss1 = F.kl_div(text_fc.log_softmax(-1), text_cf.softmax(-1), reduction='batchmean')

                term1 = torch.norm(torch.matmul(text_atn_feature, text_fc.t()), p='fro')
                term2 = torch.norm(torch.matmul(clip_feature, text_fc.t()), p='fro')
                term3 = torch.norm(torch.matmul(clip_feature, text_cf.t()), p='fro')
                term4 = torch.norm(torch.matmul(text_atn_feature, text_cf.t()), p='fro')
                loss2 = (term1 + term2 + term3 + term4) / (self.batchSize ** 2)

                loss3 = loss_fn(final_label, label.float())
                loss = loss1 + loss2 + loss3 + 0.1 * loss0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results0 = self.test(self.val_loader)
            mark = recorder.add(results0)

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write("\n")
                f.write("epoch={}, loss={}\n".format(epoch + 1, avg_loss.item()))
                for key, value in results0.items():
                    f.write(f"{key}: {value}\n")

            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir,
                                        '{}.pkl'.format(self.model_name + self.dataset + '_' + str(i))))
                torch.save(self.model, os.path.join(self.save_param_dir,
                                                    '{}.pth'.format(self.model_name + self.dataset + '_' + str(i))))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.save_param_dir, '{}.pkl'.format(self.model_name + self.dataset + '_' + str(i)))))
        print("开始进行最后的测试")
        results0 = self.test(self.test_loader)
        print("final: ")
        highlight_keys = {'auc', 'metrics', 'recall', 'precision', 'acc'}
        for key, value in results0.items():
            if key in highlight_keys:
                print(f"{key}: \033[1;32m{value}\033[0m")
            else:
                print(f"{key}: {value}")

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write("\nfinal:   lr = {}\n".format(self.lr))
            for key, value in results0.items():
                f.write(f"{key}: {value}\n")
        return

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        for step_n, batch in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                batch_data = clip_data2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred, _, _, _, _, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res
