# Mini GPT From Scratch

A minimal implementation of a GPT-style Transformer language model built entirely from scratch using **PyTorch**.
This project is designed as an educational implementation to understand how modern Large Language Models work internally.

Instead of relying on high-level libraries, the goal of this repository is to clearly demonstrate the following components:

* Token Embeddings
* Positional Embeddings
* Self-Attention Mechanism
* Multi-Head Attention
* Transformer Blocks
* Cross Entropy Loss
* Autoregressive Text Generation
* Training Loop Design
* Model Checkpointing
* Experiment Logging

This repository is ideal for developers who want to understand the internal architecture of transformer-based models before using large frameworks.

---

# Project Structure

```
mini-gpt-from-scratch/
│
├── README.md
├── main.py
├── model.py
├── train.py
├── generate.py
├── config.py
│
├── dataset/
│   └── input.txt
│
├── checkpoints/
│
└── logs/
```

### File Description

| File                | Description                                        |
| ------------------- | -------------------------------------------------- |
| `main.py`           | Entry point that connects training and generation  |
| `model.py`          | Implementation of the GPT transformer architecture |
| `train.py`          | Training loop, batching logic, checkpoint saving   |
| `generate.py`       | Autoregressive text generation                     |
| `config.py`         | Centralized hyperparameter configuration           |
| `dataset/input.txt` | Training dataset                                   |
| `checkpoints/`      | Saved model weights during training                |
| `logs/`             | Training statistics and loss logs                  |

---

# Transformer Architecture Overview

The model follows a simplified GPT architecture consisting of several transformer blocks.

```
Input Text
    ↓
Tokenization
    ↓
Token Embeddings
    ↓
Positional Embeddings
    ↓
Transformer Blocks
    ↓
Layer Normalization
    ↓
Linear Projection
    ↓
Next Token Prediction
```

Each transformer block contains:

* Multi-Head Self Attention
* Feed Forward Network
* Layer Normalization
* Residual Connections

---

# Self Attention Mechanism

The core component of a transformer is the **Scaled Dot-Product Attention**.

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

Where:

* **Q (Query)** represents the token requesting information
* **K (Key)** represents tokens being compared
* **V (Value)** represents the information passed forward
* **dₖ** is the dimension of the key vectors

This mechanism allows the model to determine which tokens in the sequence are most relevant to each other.

---

# Training Objective

The model is trained using **Next Token Prediction**.

Example:

Input sequence

```
hello worl
```

Target sequence

```
ello world
```

The model learns to predict the next token given a sequence of previous tokens.

---

# Configuration

All model and training parameters are defined in `config.py`.

Example configuration:

```
batch_size = 32
block_size = 64
max_steps = 5000

learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4

log_interval = 100
checkpoint_interval = 1000
```

This allows easy experimentation with different architectures.

---

# Requirements

```
python >= 3.9
torch
matplotlib (optional for loss plotting)
```

Install dependencies:

```
pip install torch matplotlib
```

---

# Training the Model

Start training by running:

```
python main.py
```

The training process will:

1. Load the dataset
2. Encode text into tokens
3. Create training batches
4. Train the transformer model
5. Print loss statistics
6. Save model checkpoints
7. Store training logs

Example training output:

```
Step 0 | Loss 4.12
Step 100 | Loss 3.18
Step 200 | Loss 2.41
Step 300 | Loss 1.95
Step 400 | Loss 1.60
```

---

# Model Checkpoints

During training, the model automatically saves checkpoints.

Example:

```
checkpoints/model_step_1000.pt
checkpoints/model_step_2000.pt
```

These files store the model weights and can be used to resume training or run inference.

---

# Training Logs

Training statistics are saved in:

```
logs/loss_history.json
```

This file stores the loss values recorded during training.

---

# Plotting the Training Curve (Optional)

You can visualize training progress using a simple plotting script.

Example Python code:

```
import json
import matplotlib.pyplot as plt

with open("logs/loss_history.json") as f:
    loss = json.load(f)

plt.plot(loss)

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.show()
```

---

# Generating Text

After training, the model can generate text using autoregressive sampling.

Run:

```
python main.py
```

Example generated output:

```
the king said that the war would never end
and the people of the kingdom gathered near
```

The model predicts the next token repeatedly to generate a sequence.

---

# Learning Goals

This repository was created to deeply understand:

* Transformer architecture
* Self-attention computation
* Language model training
* Token embeddings
* Autoregressive generation
* Experiment tracking and logging

The project prioritizes **clarity and educational value** rather than production-level performance.

---

# Possible Future Improvements

Several enhancements can be added to extend this project:

* Byte Pair Encoding (BPE) tokenizer
* Larger datasets
* Dropout regularization
* Mixed precision training
* Distributed training
* Temperature-based sampling
* Top-k / nucleus sampling
* Model evaluation metrics

---

# References

Important research that inspired this project:

* Attention Is All You Need
* GPT Architecture
* Transformer Language Models

---

# Contact

Madan Singh – madanrana964@gmail.com GitHub: https://github.com/madanrana

---

# License

This project is open source and intended for educational and research purposes.
