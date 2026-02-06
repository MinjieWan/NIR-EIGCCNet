# DSTC-SRNet
A Multi-Frame Super-Resolution Method Based on Deep Spatio-Temporal Collaboration 
# Abstract
Single-image super-resolution (SISR) methods rely on learning image priors in order to add high frequency details, resulting in poor algorithmic robustness. Existing multi-frame super-resolution (MFSR) methods reconstruct high frequency details by fusing sub-pixel displacement information from multiple frames. However, in complex motion and realistic scenarios, they often struggle to fully extract and effectively fuse complementary latent information within the sequence, leading to issues such as blurred details and ghosting in the reconstructed images. Therefore, in this paper, we propose an MFSR reconstruction method based on deep spatiotemporal collaboration, namely DSTC-SRNet. We employ a collaborative reconstruction framework, which is centered on temporal memory, spatial perception, and adaptive fusion. The aim of this design is to facilitate the comprehensive and orderly utilization of spatio-temporal information. First, we design a dual-stream framework that decouples information streams into parallel multi-scale temporal memory stream (MTMS) and dynamic receptive-field spatial stream (DRSS) for independent processing. Second, a novel bidirectional cross-attention fusion (BCAF) module is developed to align and merge features to generate a deep feature representation. Third, a multi-level feature enhancement (MLFE) module is further presented to refine the fused feature representation. Finally, a progressive decoder is utilized to decode the refined feature representation to produce high-resolution (HR) images. Experimental results on the BSR public dataset demonstrate that for 4× super-resolution reconstruction tasks, our method outperforms the existing methods by 5.03% and 21.9% in terms of peak signal-to-noise ratio (PSNR) and learned perceptual image patch similarity (LPIPS), respectively.

##  Project Structure

The project directory is organized as follows :

```text
NIR_EGCC_CODE/
├─ checkpoints/               # Saved weights 
├─ dataset/                   # Saved weights 
├── model/                    # Model architecture
├── support/                  # Downsample for SBSR
├── results/                  # Inference results 
├── metrics.py                # PSNR, SSIM and LPIPS
├── config.py                 # Setting
├── dataset.py                # Data loading logic
├── loss.py                   # Loss functions
├── utils.py                  # Helper functions
├── train.py                  # training script
└── test.py                   # Evaluation script
```

# Installation

This repository is built in PyTorch 2.0.0 (Python3.10, CUDA12.6).
Follow these intructions

1. Make conda environment
```
conda create -n pytorch2.0 python=3.10
conda activate pytorch2.0
```

2. Install dependencies
```
conda install pytorch=2.0 torchvision -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python tqdm
pip install numpy scipy lpips pillow
```

# Dataset
We provide the dataset of our work. You can download it by the following link: [dataset](https://pan.baidu.com/s/1jQBMkcJPPafd8hDNj323CQ?pwd=3114 ) code:3114.

# Usage

Download and release DTSC.zip. <br>
Put the dataset in the dataset folder. <br>
Setup the required parameters. <br>
Run main.py for training or testing.

