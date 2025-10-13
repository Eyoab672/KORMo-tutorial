# KORMo-tutorial

## [🤗 Model and Dataset](https://huggingface.co/KORMo-Team)
## [📄 Paper](https://arxiv.org/abs/2510.09426)

This repository provides tutorial materials for **KORMo(Korean Open Reasoning Model)**, a Korean Large Language Model (LLM) project built with the Hugging Face ecosystem.  
It demonstrates how to **pretrain**, **fine-tune**, and **evaluate** large-scale language models using modern open-source frameworks.

---

### 🧩 Setup Environment
```bash
bash setup/create_uv_venv.sh
```

This script creates an isolated virtual environment and installs all dependencies required to run the tutorials.

### 📘 Tutorials Included

You can find step-by-step examples in the `tutorial` directory:
```graphql
tutorial
  ├── 01.pretrain_from_scratch.ipynb     # Pretraining a language model from scratch using custom data
  ├── 02.sft_qlora.ipynb                 # Supervised Fine-Tuning with QLoRA for efficiency
  └── 03.inference.ipynb                 # Performing inference and evaluating the trained model
```

Each notebook is designed to be self-contained and runnable within the prepared environment.

### 🚀 Overview

These tutorials aim to help researchers and practitioners:

- Understand the full training pipeline of large Korean language models
- Learn how to use Hugging Face Transformers, Datasets, and PEFT (Parameter-Efficient Fine-Tuning)
- Experiment with QLoRA and distributed training setups
- Run inference and evaluation on trained checkpoints

### 🧠 Credits 
Developed by the KORMo Team.
