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
    ├── clip_cn_vit-b-16.pt
    ├── mae_pretrain_vit_base.pth
    ├── main.py
    ├── requirements.txt
    ├── data/
    │   ├── en/
    │   ├── pheme/
    │   ├── twitter/
    │   ├── weibo/
    │   └── weibo21/
    ├── downloadModel/
    │   ├── chinese_roberta_wwm_base_ext_pytorch/
    │   ├── clip-vit-base-patch16/
    │   ├── roberta-base/
    │   └── w2v/
    ├── downloads/
    │   └── models--roberta-base/
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
