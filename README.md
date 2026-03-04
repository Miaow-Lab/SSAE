<div align="center">

# Step-Level Sparse Autoencoder for Reasoning Process Interpretation

[![arXiv](https://img.shields.io/badge/arXiv-2603.03031-b31b1b.svg)](https://arxiv.org/abs/2603.03031)
[![datasets](https://img.shields.io/badge/datasets-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Miaow-Lab/SSAE-Dataset)
[![model](https://img.shields.io/badge/model-blue?logo=huggingface&logoColor=white)](https://huggingface.co/Miaow-Lab/SSAE-Checkpoints)

</div>

This repository contains the official implementation of the paper "Step-Level Sparse Autoencoder for Reasoning Process Interpretation"

## 🔧 Installation
### Prerequisites
- Python 3.10
- CUDA 11.8+

```bash
# Clone the repository
git clone https://github.com/Miaow-Lab/SSAE.git
cd SSAE

# Create conda environment
conda create -n ssae python=3.10
conda activate ssae

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The dataset is hosted on [HuggingFace Dataset](https://huggingface.co/Miaow-Lab/SSAE-Checkpoints). There should be 6 files to download: `gsm8k_385K_train.json`, `gsm8k_385K_valid.json`, `numina_859K_train.json`, `numina_859K_valid.json`, `opencodeinstruct_36K_train.json`, and `opencodeinstruct_36K_valid.json`.

Please create a `data/` folder at the repository root and place all downloaded files there.

```bash
mkdir -p data
# put downloaded dataset files into ./data
```

### Pretrained SSAE Checkpoints
We also provide pretrained SSAE checkpoints on [HuggingFace](https://huggingface.co/Miaow-Lab/SSAE-Checkpoints).

## ⚙️ Configuration

The project uses modular YAML configuration files located in `configs/`. You can modify parameters directly in the YAML files or override them via command line arguments.

### Global Config Structure
There are 3 main configuration files:

- **`configs/train.yaml`**: Main configuration for training SSAE models.

- **`configs/classifier.yaml`**: Configuration for the classifier pipeline (data generation, training, evaluation).

- **`configs/experiment.yaml`**: Configuration for analysis and probing experiments.

All scripts support overriding config parameters using the `--set KEY=VALUE` argument without modifying the YAML file.

## 🧰 Usage
### 1. Training SSAE
Main training entry is `train.py`.

Single-GPU training:

```bash
bash scripts/train_single.sh configs/train.yaml
```

Multi-GPU training (DDP):

```bash
NPROC_PER_NODE=<num_gpus> bash scripts/train_ddp.sh configs/train.yaml
```

If you use the provided checkpoints, please place them in your configured `model_dir`.

### 2. Classifier
We train a series of classifiers to investigate the expressiveness of SSAE.
The Classifier module covers data generation, training, and evaluation.

For full usage details, see:

- [classifier/README.md](classifier/README.md)

### 3. N2G Pattern Mining

N2G pattern mining:

```bash
bash scripts/run_n2g.sh configs/experiment.yaml
```

N2G pattern analysis/labeling:

```bash
bash scripts/run_n2g_analysis.sh configs/experiment.yaml
```

### 4. Probing Guided Weighted Voting

Run probing experiment:

```bash
bash scripts/run_probing.sh configs/experiment.yaml
```

## 📧 Contact
For questions or feedback, please contact [Xuan Yang](mailto:xyang753-c@my.cityu.edu.hk)

## 🖊 Citation
If you find this work helpful, please cite our paper:
```bibtex
@misc{yang2026steplevelsparseautoencoderreasoning,
      title={Step-Level Sparse Autoencoder for Reasoning Process Interpretation}, 
      author={Xuan Yang and Jiayu Liu and Yuhang Lai and Hao Xu and Zhenya Huang and Ning Miao},
      year={2026},
      eprint={2603.03031},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03031}, 
}
```



