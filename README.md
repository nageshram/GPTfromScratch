# GPT from Scratch 

A character-level Generative Pre-trained Transformer (GPT) language model built from scratch in PyTorch, inspired by the *"Attention Is All You Need"* paper and Andrej Karpathy's *Neural Networks: Zero to Hero* lecture series.

---

##  Features

- **Decoder-Only Transformer Architecture**: Includes token embeddings, learned positional embeddings, multi-head causal self-attention, feed-forward networks with residual connections, layer normalization, and dropout regularization.
- **Hardware Accelerated**: Automatically detects and leverages **NVIDIA CUDA** or **Apple Silicon GPU (MPS)** when available, falling back to CPU.
- **Loss Estimation**: Periodically computes and reports both **training loss** and **validation loss** without gradient computation overhead.
- **Checkpointing & Reproducibility**: Automatically saves model weights, vocabulary dictionaries (`stoi`, `itos`), and architectural hyperparameters to `model.pkl` after training.
- **Standalone Inference**: Includes a dedicated [`sample.py`](sample.py) script to load saved checkpoints and generate text instantly.

---

## Model Architecture & Hyperparameters

| Hyperparameter | Description | Value |
| :--- | :--- | :--- |
| `vocab_size` | Unique character vocabulary size | Automatically derived from dataset |
| `block_size` | Maximum sequence context length | `256` tokens |
| `batch_size` | Independent sequences processed in parallel | `64` |
| `n_embed` | Model embedding dimension | `384` |
| `num_heads` | Number of self-attention heads | `4` |
| `head_size` | Dimension of each attention head (`n_embed // num_heads`) | `96` |
| `n_layer` | Number of Transformer blocks stacked | `6` |
| `dropout_rate` | Dropout probability for regularization | `0.2` (20%) |
| `learning_rate` | AdamW optimizer learning rate | `1e-3` |
| `max_iters` | Total training iterations | `10000` |

---

## 📁 Repository Structure

```text
GPTfromScratch/
├── input.txt                              # Character-level training dataset
├── bigram.py                              # Main training script & model definition
├── sample.py                             # Inference script to generate text from model.pkl
├── model.pkl                              # Saved model weights & metadata (generated after training)
├── requirements.txt                       # Project dependencies
└── NIPS-2017-attention-is-all-you-need-Paper.pdf # Transformer reference paper
```

---

##  Quick Start

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Training the Model

Run [`bigram.py`](bigram.py) to train the model on `input.txt`. The script will output training/validation losses and save the trained checkpoint to `model.pkl`:

```bash
python bigram.py
```

#### Example Output:
```text
step 0: train loss 4.2092, val loss 4.2090
step 500: train loss 2.6929, val loss 2.6638
step 1000: train loss 2.4747, val loss 2.4752
...
step 10000: train loss 1.4820, val loss 1.6910

--- Generating text from trained model ---
...

Model saved successfully to model.pkl
```

---

### 3. Generating Text (Inference)

To generate text using your trained checkpoint without re-running training, execute [`sample.py`](sample.py):

```bash
python sample.py
```

---

##  References

- [Attention Is All You Need (Vaswani et al., 2017)](NIPS-2017-attention-is-all-you-need-Paper.pdf)
- [Andrej Karpathy: Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)
