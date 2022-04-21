Variational DMGNN GAN
==============================

Variational seq2seq DMGNN-based GAN model for human skeletal motion prediction. Course project for EECE 571F: Deep Learning with Structures at UBC, 2021 Winter Term 2.

Refer to here for [the readme file](README_DMGNN.md) from the [original DMGNN paper](https://arxiv.org/abs/2003.08802).

# Authors
[Anushree Bannadabhavi*](https://www.linkedin.com/in/anushree-bannadabhavi-585435122/?originalSubdomain=ca), [Guanxiong Chen*](https://www.linkedin.com/in/guanxiongchen/), [Yunpeng (Larry) Liu*](https://www.linkedin.com/in/larry-liu-323b51126/), [Kaitai (Alan) Tong*](https://www.linkedin.com/in/alan-tong/).

* Authors listed by alphabetical order of last names. Equal contribution from all.

# Instruction for training our GAN model
### CMU dataset random masking
```
cd v-dmgnn-gan_cmu
python main.py prediction -c ../config/CMU/v-dmgnn-gan/train_random.yaml
```
### ACCAD dataset lower body masking
```
cd v-dmgnn-gan_amass
python main.py prediction -c config/ACCAD/v-dmgnn-gan/train_lower-body.yaml
```

# Instruction for evaluating our GAN model
### CMU dataset random masking
```
cd v-dmgnn-gan_cmu
python main.py prediction -c ../config/CMU/v-dmgnn-gan/test_random.yaml
```
### ACCAD dataset lower body masking
```
cd v-dmgnn-gan_amass
python main.py prediction -c config/ACCAD/v-dmgnn-gan/test_lower-body.yaml
```
