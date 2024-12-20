# Continuous Knowledge-Preserving Decomposition with Adaptive Layer Selection for Few-Shot Class-Incremental Learning

This is the official repository for "Continuous Knowledge-Preserving Decomposition with Adaptive Layer Selection for Few-Shot Class-Incremental Learning".

> **Continuous Knowledge-Preserving Decomposition with Adaptive Layer Selection for Few-Shot Class-Incremental Learning [PDF](https://arxiv.org/abs/2501.05017)**<br>
> [Xiaojie Li](https://xiaojieli0903.github.io/)^12, [Jianlong Wu](https://wujianlong.hit.github.io)^1, [Yue Yu](https://yuyue.github.io/)^2, [Liqiang Nie](https://liqiangnie.github.io/)^1, [Min Zhang](https://zhangmin2021.hit.github.io)^1<br>
> ^1Harbin Institute of Technology, Shenzhen ^2Peng Cheng Laboratory

![CKPD-FSCIL Framework](figs/framework.png)

## 🔨 Installation

1. **Create Conda environment**:

   ```bash
   conda create --name ckpdfscil python=3.10 -y
   conda activate ckpdfscil
   ```

2. **Install dependencies**:

   ```bash
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   pip install -U openmim
   mim install mmcv-full==1.7.0 mmengine==0.10.4
   pip install opencv-python matplotlib einops timm==0.6.12 scikit-learn transformers==4.44.2
   pip install git+https://github.com/openai/CLIP.git
   git clone https://github.com/state-spaces/mamba.git && cd mamba && git checkout v1.2.0.post1 && pip install .
   ```

3. **Clone the repository**:

   ```bash
   git clone https://github.com/xiaojieli0903/CKPD-FSCIL.git
   cd CKPD_FSCIL && mkdir ./data
   ```

---

## ➡️ Data Preparation

1. **Download datasets** from [NC-FSCIL link](https://huggingface.co/datasets/HarborYuan/Few-Shot-Class-Incremental-Learning/blob/main/fscil.zip).

2. **Organize the datasets**:

   ```bash
   ./data/
   ├── cifar/
   ├── CUB_200_2011/
   └── miniimagenet/
   ```

---

## ➡️ Pretrained Models Preparation

Use `tools/convert_pretrained_model.py` to convert models. Supported types:

- **CLIP**: Converts OpenAI CLIP models.
- **TIMM**: Converts TIMM models.

### Commands

- **CLIP Model**:

  ```bash
  python tools/convert_pretrained_model.py ViT-B/32 ./pretrained_models/clip-vit-base-p32_openai.pth --model-type clip
  ```

- **TIMM Model**:

  ```bash
  python tools/convert_pretrained_model.py vit_base_patch16_224 ./pretrained_models/vit_base_patch16_224.pth --model-type timm
  ```

---


## 🚀 Training
Execute the provided scripts to start training:

### Mini Imagenet
```commandline
sh train_miniimagenet.sh
```

### CUB
```commandline
sh train_cub.sh
```

## ✏️ Citation
If you find our work useful in your research, please consider citing:
```
@article{li2025continuous,
  title={[Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/abs/2501.05017)},
  author={Li, Xiaojie and Wu, Jianlong and Yu, Yue and Nie, Liqiang and Zhang, Min},
  journal={arXiv preprint arXiv:2501.05017},
  year={2025}
}
```
