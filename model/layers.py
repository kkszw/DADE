# author: szw
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score


class MSRA(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.conv_group = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat_dim, feat_dim // 4, kernel_size=2 * i + 3, padding=i + 1),
                nn.ReLU(),
                nn.BatchNorm1d(feat_dim // 4)
            ) for i in range(3)
        ])
        # self.global_branch = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),  # [B,512,L] -> [B,512,1]
        #     nn.Conv1d(embed_dim, embed_dim // reduction_ratio, 1),  # Point-wise
        #     nn.BatchNorm1d(embed_dim // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Conv1d(embed_dim // reduction_ratio, embed_dim, 1),  # Point-wise
        #     nn.BatchNorm1d(embed_dim),
        # )

        self.fusionX = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        self.fusionY = nn.Sequential(
            nn.Linear(384, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, feat_dim)
        )

        self.modal_gate = MLP(2 * feat_dim, [512], 0.2)

    def forward(self, X, Y):
        X_1d = X
        Y_1d = Y
        for i in range(2):
            tempX = X_1d.unsqueeze(-1)  # [64,512,1]
            tempY = Y_1d.unsqueeze(-1)

            x_feats = []
            for conv in self.conv_group:
                feat = conv(tempX).squeeze(-1)  # [64,128]
                x_feats.append(feat)

            y_feats = []  # [64, 384]
            for conv in self.conv_group:
                feat = conv(tempY).squeeze(-1)  # [64,128]
                y_feats.append(feat)

            fused_x = torch.cat(x_feats, dim=-1)  # [64, 384]
            fused_y = torch.cat(y_feats, dim=-1)

            enhanced_X = X_1d + self.fusionX(fused_x)  # [64, 512]
            enhanced_Y = Y_1d + self.fusionY(fused_y)
            gate = torch.sigmoid(self.modal_gate(torch.cat([enhanced_X, enhanced_Y], dim=-1)))
            X_1d = gate * X
            Y_1d = (1 - gate) * Y
        return X_1d + Y_1d


class MSRAA(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusio_layer = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 16, 1),
            nn.BatchNorm1d(feat_dim // 16),
            nn.ReLU(),
            nn.Conv1d(feat_dim // 16, feat_dim, 1),  # Point-wise
            nn.BatchNorm1d(feat_dim),
        )

        self.modal_gate = MLP(feat_dim, [512], 0.2)

    def forward(self, X, Y):
        X_1d = X
        Y_1d = Y
        for i in range(2):
            tempX = X_1d.unsqueeze(-1)  # [64,512,1]
            tempY = Y_1d.unsqueeze(-1)

            x_feats = self.fusio_layer(tempX).squeeze(-1)  # [64,512]
            y_feats = self.fusio_layer(tempY).squeeze(-1)
            gate = torch.sigmoid(x_feats + y_feats)
            X_1d = gate * X
            Y_1d = (1 - gate) * Y
        return X_1d + Y_1d


class TextGranularityConstraints(nn.Module):
    def __init__(self, device='cuda'):
        super(TextGranularityConstraints, self).__init__()
        self.device = device

    def gaussian(self, T_pos, T_neg, sigma=1.0):
        """计算高斯核矩阵（MMD用）"""
        K_XX = torch.exp(-torch.sum((T_pos.unsqueeze(1) - T_pos.unsqueeze(0)) ** 2, dim=-1) / (2 * sigma ** 2)).mean()
        K_XY = torch.exp(-torch.sum((T_pos.unsqueeze(1) - T_neg.unsqueeze(0)) ** 2, dim=-1) / (2 * sigma ** 2)).mean()
        K_YY = torch.exp(-torch.sum((T_neg.unsqueeze(0) - T_neg.unsqueeze(0)) ** 2, dim=-1) / (2 * sigma ** 2)).mean()
        return K_XX - 2 * K_XY + K_YY

    def forward(self, T_prime, labels):
        """
        参数:
            T_prime: 文本特征 [B, N]
            labels: 新闻标签 [B]
        """
        T_pos = T_prime[labels == 1]
        T_neg = T_prime[labels == 0]

        mmd = self.gaussian(T_pos, T_neg, sigma=1.0)

        l_t = torch.tensor(0.0, device=self.device)
        l_f = torch.tensor(0.0, device=self.device)

        if len(T_pos) > 1:
            sim_matrix = F.cosine_similarity(T_pos.unsqueeze(1), T_pos.unsqueeze(0), dim=2)
            mask = ~torch.eye(len(T_pos), dtype=torch.bool, device=self.device)
            l_t += (1 - sim_matrix[mask]).mean()  # 最小化真实新闻间的差异

        if len(T_neg) > 1:
            sim_matrix = F.cosine_similarity(T_neg.unsqueeze(1), T_neg.unsqueeze(0), dim=2)
            mask = ~torch.eye(len(T_neg), dtype=torch.bool, device=self.device)
            l_f += (1 - sim_matrix[mask]).mean()  # 最小化虚假新闻间的差异
        return l_t + l_f - 0.1 * mmd


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


def add_gaussian_noise(features, noise_std=0.3):
    noise = torch.randn_like(features) * noise_std
    return features + noise


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class MLP_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MaskAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim, 1)

    def forward(self, inputs, mask):
        score = self.Line(inputs).view(-1, inputs.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score, inputs).squeeze(1)
        return output


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v += (x - self.v) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Recorder:
    def __init__(self, early_step):
        self.max = {'metrics': 0}
        self.current = {'metrics': 0}
        self.max_index = 0
        self.current_index = 0
        self.early_step = early_step

    def add(self, x):
        self.current = x
        self.current_index += 1
        print("current: ")
        for key, value in self.current.items():
            print(f"{key}: {value}")
        return self.judge()

    def judge(self):
        if self.current['metrics'] > self.max['metrics']:
            self.max = self.current
            self.max_index = self.current_index
            self.showFinal()
            return 'save'
        # self.showFinal()
        if self.current_index - self.max_index >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showFinal(self):
        print("Max: ")
        for key, value in self.max.items():
            print(f"{key}: {value}")


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


def metricsTrueFalse(y_true, y_pred, category, category_dict):
    y_GT = y_true
    metrics_true_false = metrics(y_true, y_pred, category, category_dict)
    fake = {}
    real = {}
    THRESH = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    real_news_TP, real_news_TN, real_news_FP, real_news_FN = [0] * 9, [0] * 9, [0] * 9, [0] * 9
    fake_news_TP, fake_news_TN, fake_news_FP, fake_news_FN = [0] * 9, [0] * 9, [0] * 9, [0] * 9
    real_news_sum, fake_news_sum = [0] * 9, [0] * 9
    for thresh_idx, thresh in enumerate(THRESH):
        for i in range(len(y_pred)):
            if y_pred[i] < thresh:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        for idx in range(len(y_pred)):
            if y_GT[idx] == 1:
                #  FAKE NEWS RESULT
                fake_news_sum[thresh_idx] += 1
                if y_pred[idx] == 0:
                    fake_news_FN[thresh_idx] += 1
                    real_news_FP[thresh_idx] += 1
                else:
                    fake_news_TP[thresh_idx] += 1
                    real_news_TN[thresh_idx] += 1
            else:
                # REAL NEWS RESULT
                real_news_sum[thresh_idx] += 1
                if y_pred[idx] == 1:
                    real_news_FN[thresh_idx] += 1
                    fake_news_FP[thresh_idx] += 1
                else:
                    real_news_TP[thresh_idx] += 1
                    fake_news_TN[thresh_idx] += 1

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0] * 9, [0] * 9, [0] * 9, [0] * 9, [
        0] * 9
    real_recall, fake_recall, real_F1, fake_F1 = [0] * 9, [0] * 9, [0] * 9, [0] * 9
    for index, _ in enumerate(THRESH):
        val_accuracy[index] = (real_news_TP[index] + real_news_TN[index]) / (
                real_news_TP[index] + real_news_TN[index] + real_news_FP[index] + real_news_FN[index])
        real_accuracy[index] = (real_news_TP[index]) / real_news_sum[index]
        fake_accuracy[index] = (fake_news_TP[index]) / fake_news_sum[index]
        real_precision[index] = real_news_TP[index] / max(1, (real_news_TP[index] + real_news_FP[index]))
        fake_precision[index] = fake_news_TP[index] / max(1, (fake_news_TP[index] + fake_news_FP[index]))
        real_recall[index] = real_news_TP[index] / max(1, (real_news_TP[index] + real_news_FN[index]))
        fake_recall[index] = fake_news_TP[index] / max(1, (fake_news_TP[index] + fake_news_FN[index]))
        real_F1[index] = 2 * (real_recall[index] * real_precision[index]) / max(1, (
                real_recall[index] + real_precision[index]))
        fake_F1[index] = 2 * (fake_recall[index] * fake_precision[index]) / max(1, (
                fake_recall[index] + fake_precision[index]))

    fake['precision'] = fake_precision[0]
    fake['recall'] = fake_recall[0]
    fake['F1'] = fake_F1[0]
    real['precision'] = real_precision[0]
    real['recall'] = real_recall[0]
    real['F1'] = real_F1[0]
    metrics_true_false['real'] = real
    metrics_true_false['fake'] = fake
    return metrics_true_false


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}

    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    # 按类别分组真实值和预测值
    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    # 计算每个类别的 AUC
    for c, res in res_by_category.items():
        try:
            auc = roc_auc_score(res['y_true'], res['y_pred'])
            metrics_by_category[c] = {
                'auc': round(auc, 4)
            }
        except ValueError:
            pass

    # 计算全局 AUC
    try:
        metrics_by_category['auc'] = round(roc_auc_score(y_true, y_pred, average='macro'), 4)
    except ValueError:
        pass

    # 计算全局指标（F1, Recall, Precision, Accuracy）
    y_pred_rounded = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metrics'] = round(f1_score(y_true, y_pred_rounded, average='macro'), 4)
    metrics_by_category['recall'] = round(recall_score(y_true, y_pred_rounded, average='macro', zero_division=0), 4)
    metrics_by_category['precision'] = round(precision_score(y_true, y_pred_rounded, average='macro', zero_division=0),
                                             4)
    metrics_by_category['acc'] = round(accuracy_score(y_true, y_pred_rounded), 4)

    # 计算每个类别的 Precision, Recall, F1, Accuracy
    for c, res in res_by_category.items():
        y_pred_rounded_cat = np.around(np.array(res['y_pred'])).astype(int)
        metrics_by_category[c] = {
            'precision': round(precision_score(res['y_true'], y_pred_rounded_cat, average='macro', zero_division=0), 4),
            'recall': round(recall_score(res['y_true'], y_pred_rounded_cat, average='macro', zero_division=0), 4),
            'f1score': round(f1_score(res['y_true'], y_pred_rounded_cat, average='macro'), 4),
            'acc': round(accuracy_score(res['y_true'], y_pred_rounded_cat), 4),
        }

    return metrics_by_category
