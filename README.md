## DADE

Code For Domain Adaptation Network with Dual-Encoder for Fake News Detection(DADE)

## Requirements

The required packages are listed in requirements.txt. 

You can install the dependencies using pip:

    pip install -r requirements.txt

## Data Preparation

1. Data Splitting: In the experiments, we maintain the same data splitting scheme as the benchmarks.
2. Weibo21 Dataset: For the Weibo21 dataset, we follow the work from [(Ying et al.， 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to [Dr. Qiong Nan](mailto:nanqiong19z@ict.ac.cn) to get the complete multimodal multi-domain dataset Weibo21.
3. Weibo Dataset: For the Weibo dataset, we adhere to the work from [(Wang et al.， 2022)](https://github.com/yaqingwang/EANN-KDD18). In addition, we have incorporated domain labels into this dataset. You can download the final processed data from the link below.
4.  By using this data, you will bypass the data preparation step. Link: https://pan.baidu.com/s/1AaDWvl_zS3omvl50bZv3UA?pwd=89mw

## Pretrained Models

1. you will bypass the pretrained models step. Link: https://pan.baidu.com/s/1AaDWvl_zS3omvl50bZv3UA?pwd=89mw

## Directory Structure

The project is organized as follows:
'''
.
├── clip_cn_vit-b-16.pt
├── mae_pretrain_vit_base.pth
├── main.py
├── requirements.txt
├── data/
│   ├── en/
│   │   ├── test.pkl
│   │   ├── train.pkl
│   │   └── val.pkl
│   ├── pheme/
│   │   ├── pheme_binary.csv
│   │   ├── pheme_test.csv
│   │   ├── pheme_train.csv
│   │   └── pheme_val.csv
│   ├── twitter/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   ├── twitter16_merged.csv
│   │   └── val.csv
│   ├── weibo/
│   │   ├── test_weibo.xlsx
│   │   ├── train_weibo.xlsx
│   │   └── val_weibo.xlsx
│   └── weibo21/
│       ├── readme
│       ├── test_2_domain.xlsx
│       ├── train_2_domain.xlsx
│       └── val_2_domain.xlsx
├── downloadModel/
│   ├── chinese_roberta_wwm_base_ext_pytorch/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── clip-vit-base-patch16/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.json
│   ├── roberta-base/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── tokenizer.json
│   │   └── vocab.json
│   └── w2v/
│       └── Tencent_AILab_Chinese_w2v_model.kv
├── downloads/
│   ├── .locks/
│   │   └── models--roberta-base/
│   └── models--roberta-base/
│       ├── blobs/
│       │   ├── 5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81
│       │   └── 8db5e7ac5bfc9ec8b613b776009300fe3685d957
│       ├── refs/
│       │   └── main
│       └── snapshots/
│           └── e2da8e2f811d1448a5b465c236feacd80ffbac7b/
│               ├── config.json
│               └── model.safetensors
├── model/
│   ├── clip_cn_vit-b-16.pt
│   ├── layers.py
│   ├── singleModel.py
│   └── t-sne.py
├── param_model/
│   └── singleModel/
│       ├── en/
│       ├── pheme/
│       ├── twitter/
│       ├── weibo/
│       └── weibo21/
└── utils/
    ├── dataLoader.py
    ├── models_mae.py
    └── pos_embed.py
'''
