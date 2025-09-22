# DADE
Code For Domain Adaptation Network with Dual-Encoder for Fake News Detection(DADE)
DADE
│  clip_cn_vit-b-16.pt
│  mae_pretrain_vit_base.pth
│  main.py
│
├─data
│  ├─weibo
│  │      test_weibo.xlsx
│  │      train_weibo.xlsx
│  │      val_weibo.xlsx
│  │
│  └─weibo21
│          readme
│          test_2_domain.xlsx
│          train_2_domain.xlsx
│          val_2_domain.xlsx
│
├─downloadModel
│  ├─chinese_roberta_wwm_base_ext_pytorch
│  ├─clip-vit-base-patch16
│  ├─roberta-base
│  └─w2v
│
├─model
│  │  layers.py
│  │  singleModel.py
│  │  t-sne.py
│
├─param_model
│  └─singleModel
│      ├─weibo
│      └─weibo21
│
├─utils
│  │  dataLoader.py
│  │  models_mae.py
│  │  pos_embed.py
