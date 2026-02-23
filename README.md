# SSAE Repository

This repository contains training and experiment code for SSAE and downstream analyses.

## 🔧 Installation
### Prerequisites
- Python 3.10
- CUDA 11.8+

```bash
# Clone the repository
git clone xxx.git
cd SSAE

# Create conda environment
conda create -n ssae python=3.10
conda activate ssae

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The dataset is hosted on Hugging Face:

- Dataset link: https://huggingface.co/datasets/<YOUR_DATASET_NAME>

There should be 6 files to download: `gsm8k_385K_train.json`, `gsm8k_385K_valid.json`, `numina_859K_train.json`, `numina_859K_valid.jsonl`, `opencodeinstruct_train.jsonl`, and `opencodeinstruct_valid.jsonl`.

Please create a `data/` folder at the repository root and place all downloaded files there.

```bash
mkdir -p data
# put downloaded dataset files into ./data
```

### Pretrained SSAE Checkpoints
We provide pretrained SSAE checkpoints on Hugging Face:
- **GSM8K Dataset**: https://huggingface.co/<YOUR_CKPT_REPO>
- **Numina Dataset**: https://huggingface.co/<YOUR_CKPT_REPO>
- **OpenCodeInstruct Dataset**: https://huggingface.co/<YOUR_CKPT_REPO>


## 🧰 Usage
### 1. Training SSAE
Main training entry is `train.py`.

Single-GPU training:

```bash
bash scripts/train_single.sh configs/train.yaml
```

Multi-GPU training (DDP):

```bash
NPROC_PER_NODE=your_gpu_node bash scripts/train_ddp.sh configs/train.yaml
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

---

All experiment/training scripts support config overrides via `--set KEY=VALUE`.

## 📧 Contact
For questions or feedback, please contact [Xuan Yang](mailto:xyang753-c@my.cityu.edu.hk)

## 🖊 Citation
If you find this work helpful, please cite our paper:
```bibtex

```



