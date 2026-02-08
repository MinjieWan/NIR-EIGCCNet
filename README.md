# NIR-EIGCCNet: A Near-Infrared Edge Information-Guided Color Constancy Method

This is the official implementation of **NIR-EIGCCNet**, a color constancy framework designed to restore RGB images by leveraging Near-Infrared edge guidance.

---

##  Project Structure

The project directory is organized as follows :

```text
NIR_EIGCC/
├─ checkpoints/               # Saved weights (.h5)
├─ data/
│   ├── train/                # Training .h5 files
│   ├── test/                 # Testing .h5 files
│   ├── train_list.txt        # Mapping for training pairs
│   └── test_list.txt         # Mapping for testing pairs
├── model/
│   └── nir_eigcc.py           # Model architecture
├── results/                  # Inference results 
├── dataset_manager.py        # Data loading logic
├── losses.py                 # Loss functions
├── utils.py                  # Helper functions
├── train_stage_1.py          # Stage 1 training script
├── train_stage_2.py          # Stage 2 fine-tuning script
└── test.py                   # Evaluation script
```

---

##  Requirements

The project is tested with the following environment:

* **Python** == 3.7.16
* **TensorFlow-GPU** == 2.7.0
* **NumPy** == 1.21.6
* **OpenCV-Python** == 4.12.0
* **scikit-image** == 0.19.3
* **h5py** == 3.8.0
* **Pandas** == 1.3.5
* **Matplotlib** == 3.5.3

---

## Data Preparation

**Source:** OMSIV subset from the [SSMID Dataset](https://xavysp.github.io/ssmid-dataset/).

**Data Format:**
* **Input**: 4-channel `.h5` files (RGB + NIR).
* **Ground Truth**: 3-channel `.h5` files (RGB).

**Mapping Lists:**
Place `train_list.txt` and `test_list.txt` in the `data/` folder. Paths must be relative to the `.txt` location:

```text
# train_list.txt example
train/input_0001.h5 train/gt_0001.h5
train/input_0002.h5 train/gt_0002.h5
……
# test_list.txt example
test/input_0001.h5 test/gt_0001.h5
……
```

---

##  Usage

### 1. Stage 1
In the first stage, the network jointly learns structural representations and coarse color restoration to establish a stable global reconstruction baseline.
```bash
python train_stage_1.py --epochs 100 --batch_size 8 --lr 1e-4
```

### 2. Stage 2
The optimization focus is shifted toward color correction by significantly increasing color-related loss weights, enabling precise color refinement while maintaining structural fidelity.
```bash
# Provide the path to your best weights from Stage 1
python train_stage_2.py --pretrained_path checkpoints/Stage_1/xxx/final_weights.h5 --epochs 50 --lr 1e-5
```

### 3. Evaluation and Inference
To evaluate the model you just trained in Stage 2, run:
```bash
# Provide the path to your Stage 2 weights
python test.py --ckpt_path checkpoints/Stage_2/xxx/finetuned_final.h5
```

---

## Pre-trained Models

Pre-trained weights are available in the `checkpoints/pretrained/` directory. 

* **Stage 1 :**`my_stage_1.h5`
* **Stage 2 :**`my_stage_2.h5`

### Inference with Pre-trained Weights
To evaluate the final model using these weights, run:
```bash
python test.py --ckpt_path checkpoints/pretrained/my_stage_2.h5
```

## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

