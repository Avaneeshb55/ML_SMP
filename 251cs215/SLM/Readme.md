# TinySLM: Pre-training a Small Language Model from Scratch

TinySLM is a custom, ~53-million parameter autoregressive GPT-style language model built entirely from scratch using PyTorch. The model is pre-trained on a subset of the **TinyStories** dataset, designed to learn coherent English grammar, basic reasoning, and narrative structure through next-token prediction.

---

##  Model Architecture & Specifications

The architecture closely follows the standard decoder-only Transformer design with casual multi-head attention and residual connections. 

* **Total Parameters:** 52,933,201
* **Vocabulary Size:** 50,257 (GPT-2 tokenization via `tiktoken`)
* **Context Window (`block_size`):** 256 tokens
* **Embedding Dimension (`n_embd`):** 384
* **Attention Heads (`n_head`):** 8 (48 dimensions per head)
* **Transformer Layers (`n_layer`):** 8
* **Optimizer:** AdamW ($lr = 1\times 10^{-3}$, decayed down to $1\times 10^{-4}$)
* **Learning Rate Scheduler:** Cosine Annealing

---

## Setup

```bash
pip install -r requirements.txt
```

---

###  Project Steps Summary

| Step | Purpose |
| :--- | :--- |
| **Step 1: Environment & Dataset** | Configures stable download timeout limits and streams the raw *TinyStories* dataset to local storage. |
| **Step 2: Training Tokenization** | Converts raw training story texts into sequences of standard GPT-2 numerical token IDs using `tiktoken`. |
| **Step 2.5: Validation Tokenization** | Processes a matching 1% validation slice into token IDs to measure true model generalization. |
| **Step 3: Memory-Mapped Data Loader** | Streams training and validation batches directly from disk via memory mapping to minimize RAM usage. |
| **Step 4: Model Configuration** | Defines the structural dimensions, network bounds, and hyperparameters for the neural layers. |
| **Step 5: Training & Checkpointing** | Optimizes model weights using AdamW and Cosine Annealing, saving the best configuration on validation drops. |
| **Step 6: Multi-Temperature Inference** | Restores weights from the best saved checkpoint to generate new text at varying levels of randomness. |

---

##  How it Runs

* **Automated Training & Checkpointing:** The script executes data processing, memory-mapping, and model training sequentially. The training loop automatically monitors validation loss and serializes the best performing model states directly to `best_tinyslm_model.pt`.
* **Inference Pipeline:** The final evaluation cell safely loads `best_tinyslm_model.pt` via `torch.serialization.add_safe_globals([GPTConfig])` to generate text loops across multiple temperature variables.

---

## Performance & Results

* Final Training Loss / Perplexity  : [2.3681 / 10.68]

* Final Validation Loss / Perplexity: [ 2.2385 / 9.38]

---

## Sample Generated Story (Prompt: "Once upon a time there was a girl" , Temperature = 0.3)

```
once upon a time there was a girl named Lily. She loved to play with her. She saw her parents and daddy all about it.

After Lily was being so excited, she decided to stay and they found a big stick. She was very competitive. She was happy. She was so happy to make it for her mom if she was so happy to her mom. She was so she had a new friends with her mommy.

Lily was too. She was very proud of the table.<|endoftext|>Once upon a

```

---