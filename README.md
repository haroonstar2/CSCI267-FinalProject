# Grid vs. Graph Representations in Self-Play Chess Reinforcement Learning

A controlled PyTorch comparison of **Mini-AlphaZero** (2D Residual CNN) and **Mini-AlphaGateau** (Directed Move-Graph Attention Network) trained *tabula rasa* via PUCT Monte Carlo Tree Search (MCTS) on 5x5 Gardner Mini-Chess.

## Overview

Deep reinforcement learning agents like AlphaZero model chess positions as uniform 2D spatial grids, requiring deep stacks of local 3x3 convolutions to propagate information across long-range sliding lines and non-local knight jumps. **AlphaGateau** (Rigaux & Kashima, NeurIPS 2024) addresses this bottleneck by representing the board as a directed move graph G(V, E): squares as nodes and geometrically valid moves as directed edges—and jointly updating node and edge features using GATEAU (Graph Attention with Edge-feature AUgmentation) layers.

This repository isolates the **2D grid vs. move-graph inductive bias** under a tightly controlled compute and parameter budget:
* **Identical Game Environment:** Custom lightweight 5x5 Gardner Mini-Chess engine (`25` board squares, `468` precomputed directed move edges).
* **Matched Capacity:** Both architectures use `4` residual blocks and nearly identical trainable parameter counts 694k parameters.
* **Shared Action Space & Search:** Both models output logits over the exact same `468` move edges and use identical standard PUCT MCTS so performance differences strictly reflect network representation rather than search-algorithm changes.

---

## Repository Structure

```text
.
├── project.ipynb                      # Main Google Colab / Jupyter notebook (environment, models, MCTS, training, evaluation)
├── README.md                          # Project documentation and reproduction guide
└── checkpoints/                       # (Generated during training) Saved PyTorch .pt model weights
    ├── minialphazero_iter{5..25}_seed42.pt
    ├── minialphazero_final_seed42.pt
    ├── minialphagateau_iter{5..25}_seed42.pt
    └── minialphagateau_final_seed42.pt
```

*(Adjust file paths above if you place `project.ipynb` or checkpoints in a different directory.)*

---

## Dependencies and Environment Setup

This project is implemented in pure Python and PyTorch, with no external chess engine libraries required.

### Software Requirements
* **Python:** `3.10+`
* **PyTorch:** `2.0+` (requires `torch.Tensor.scatter_reduce_` with `reduce="amax"`, introduced in PyTorch 1.12+)
* **NumPy:** `1.24+`
* **Matplotlib:** `3.7+`
* **Jupyter / Google Colab** (to execute `project.ipynb`)

Standard library modules used: `math`, `random`, `time`, `collections`, and `typing`.

### Hardware Requirements
* Single Google Colab NVIDIA T4 GPU.
* The notebook automatically detects CUDA. Running the full 25-iteration experiment on CPU is possible but will be significantly slower.

### Local Installation
If running locally instead of Google Colab:

```bash
pip install torch numpy matplotlib jupyter
jupyter notebook project.ipynb
```

---

## Data Access, External Files, and Model Weights

* All training data are generated dynamically *tabula rasa* through self-play games guided by PUCT MCTS. Training tuples are stored in an in-memory sliding FIFO replay buffer.
* Both `MiniAlphaZero` and `MiniAlphaGateau` are initialized from scratch using PyTorch's default initialization under `seed = 42`.
* The evaluation functions require saved `.pt` state-dictionary files produced by `train_model(...)` in the working directory:
  * `minialphagateau_final_seed42.pt` (Iteration 25 `MiniAlphaGateau`)
  * `minialphagateau_iter10_seed42.pt` (Iteration 10 `MiniAlphaGateau`)
  * `minialphazero_final_seed42.pt` (Iteration 25 `MiniAlphaZero`)
  * *(Optional)* If you wish to skip the approx 3.3-hour training run, upload/place pre-saved `.pt` checkpoints into the notebook working directory before running the evaluation cells.
---

## Key Settings and Hyperparameters

### Environment & State Encoding
| Component | Setting | Description |
| :--- | :--- | :--- |
| Game Variant | 5x5 Gardner Mini-Chess | Canonical board orientation (current player always moves row 0 to 4; no 2-square pawn advance, castling, or *en passant* |
| Move Graph | `25` nodes, `468` directed edges | All geometrically valid queen-like slides (1–4 steps), knight jumps, and rank 3 to 4 pawn promotions (`N, B, R, Q`) |
| Max Game Length | `max_moves = 80` | Games reaching 80 plies without checkmate/stalemate terminate as a draw |
| Node Features | `NODE_FEAT_DIM = 14` | 12 one-hot piece channels (6 current player, 6 opponent), 1 color plane, 1 normalized move count |
| Edge Features | `EDGE_FEAT_DIM = 7` | 1 dynamic legal-move flag, 2 normalized displacements, 4 one-hot promotion channels |

### Model Configurations
| Model | Class | Residual Blocks | Hidden Dim | Trainable Parameters |
| :--- | :--- | :---: | :---: | :---: |
| **Mini-AlphaZero (CNN)** | `MiniAlphaZero` | `4` (`ResBlockCNN`) | `64` | `694,565` | 
| **Mini-AlphaGateau (GNN)** | `MiniAlphaGateau` | `4` (`ResGATEAUBlock`) | `117` | `693,344` |

### Self-Play MCTS & Training Configuration
| Parameter | Value |
| :--- | :--- | 
| Search Algorithm | PUCT MCTS | 
| Root Exploration Noise | Dirichlet ($\alpha = 0.3$) | 
| Move Sampling Schedule | `temperature_moves = 10` | 
| Main Training Budget | `num_iters = 25` | 
| Replay Buffer | `window_size = 5000` | 
| Optimization | Adam (`batch_size = 64`) | 
| Loss Function | AlphaZero Joint Loss | 
| Checkpointing & Seed | `checkpoint_freq = 5`, `seed = 42` |

---

## How to Run (Reproducing Reported Results)

Open `project.ipynb` in Google Colab (select **Runtime -> Change runtime type -> T4 GPU**) or a local CUDA-enabled Jupyter environment, and run the cells sequentially.

### 1. Initialize Environment, Graph, and Models
Run Cells 1–7 to:
1. Import dependencies and configure `device` (`cuda`).
2. Precompute the 5x5 move graph (`25` nodes, `468` edges).
3. Define `GardnerChess`, `encode_state`, `MiniAlphaZero`, `MiniAlphaGateau`, and the PUCT MCTS self-play routines.
4. Instantiate parameter-matched models and verify counts:
```python
az_model = MiniAlphaZero(hidden_dim=64, num_blocks=4).to(device)      # 694,565 params
ag_model = MiniAlphaGateau(hidden_dim=117, num_blocks=4).to(device)   # 693,344 params
```

### 2. Quick Sanity Check
Cell 9 runs a fast 3-iteration verification (~3 minutes total on a T4 GPU):
```python
az_metrics = train_model(az_model, "MiniAlphaZero", num_iters=3, games_per_iter=4, num_simulations=16)
ag_metrics = train_model(ag_model, "MiniAlphaGateau", num_iters=3, games_per_iter=4, num_simulations=16)
show_results(az_metrics, ag_metrics, "Quick Test with num_iters=3, games_per_iter=4, num_simulations=16")
```

### 3. Run the Main Training Loop (25 Iterations)
Run Cell 10 (~63 min for `MiniAlphaZero`, ~134 min for `MiniAlphaGateau` on a T4 GPU). This logs per-iteration losses/timings, saves periodic checkpoints (`iter5` through `iter25` and `final`), and plots Total Loss vs. Iteration and Total Loss vs. Wall-Clock Time:
```python
az_metrics = train_model(az_model, "MiniAlphaZero", num_iters=25, games_per_iter=16, num_simulations=32)
ag_metrics = train_model(ag_model, "MiniAlphaGateau", num_iters=25, games_per_iter=16, num_simulations=32)

show_results(az_metrics, ag_metrics, "Main Experiment with num_iters=25, games_per_iter=16, num_simulations=32")
```

### 4. Run Head-to-Head Evaluations
All evaluation matches use greedy PUCT MCTS (`num_simulations=16`, `add_dirichlet_noise=False`) over 100 games with balanced color assignments.

**A. Deterministic Starting Position Match (Cell 11 — Illustrates Repetition Artifact):**
```python
w, d, l = play_match(
    "minialphagateau_final_seed42.pt",
    "minialphazero_final_seed42.pt",
    num_games=100,
    num_simulations=16,
)
print(f"Loaded MiniAlphaGateau vs Loaded MiniAlphaZero: {w}W / {d}D / {l}L")
```

**B. Paired Diverse-Opening Match: Final vs. Final:**
Samples 50 distinct random 2-ply openings and plays each opening twice with colors swapped (100 games total):
```python
stats = play_match_diverse(
    "minialphagateau_final_seed42.pt",
    "minialphazero_final_seed42.pt",
    num_games=100,
    num_simulations=16,
    random_opening_plies=2,
)
```

**C. Wall-Clock Controlled Match: Gateau Iter 10 vs. AlphaZero Final:**
Tests `MiniAlphaGateau` at Iteration 10 (`3,066.8 s` elapsed) against fully trained `MiniAlphaZero` at Iteration 25 (`3,758.9 s` elapsed):
```python
stats = play_match_diverse(
    "minialphagateau_iter10_seed42.pt",
    "minialphazero_final_seed42.pt",
    num_games=100,
    num_simulations=16,
    random_opening_plies=2,
)
```

---

## Reported Results Summary

### Training Metrics (25 Iterations, Google Colab T4 GPU)
| Model | Parameters | Total Training Time | Avg. Time / Iter | Best Total Loss (Iter) | Final Total Loss | Final Policy Loss | Final Value Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini-AlphaZero (CNN)** | `694,565` | `3,758.9 s` | `150.4 s` | `1.7892` (Iter 10) | `1.9807` | `1.8000` | `0.1807` |
| **Mini-AlphaGateau (GNN)** | `693,344` | `8,068.8 s` | `322.8 s` | **`1.7348`** (Iter 24) | **`1.7451`** | **`1.5853`** | **`0.1598`** |

### Head-to-Head Playing Strength (100 Games, 16 Greedy MCTS Simulations/Move)
Results are reported from **Mini-AlphaGateau's** perspective (Win Score $= W + 0.5D$):

| Matchup | Protocol | Total (W / D / L) | Win Score | As White (50 Games) | As Black (50 Games) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AG Final (Iter 25) vs. AZ Final (Iter 25)** | Deterministic start (`play_match`) | `50W / 50D / 0L` | `75.0%`* | — | — |
| **AG Final (Iter 25) vs. AZ Final (Iter 25)** | 50 paired 2-ply openings (`play_match_diverse`) | **`79W / 19D / 2L`** | **`88.5%`** | `37W / 11D / 2L` (`85.0%`) | `42W / 8D / 0L` (`92.0%`) |
| **AG Iter 10 vs. AZ Final (Iter 25)** | 50 paired 2-ply openings (`play_match_diverse`) | **`59W / 41D / 0L`** | **`79.5%`** | `33W / 17D / 0L` (`83.0%`) | `26W / 24D / 0L` (`76.0%`) |
---

## Code Attribution and References

All code in `project.ipynb` is an original, standalone **pure PyTorch** reimplementation written for this study. Rather than directly importing external codebases, it adapts the core algorithms, architectural equations, and experimental design from the following foundational and recent works:

* **AlphaZero Self-Play RL & Residual CNN Architecture:**
  Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., & Hassabis, D. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. arXiv:1712.01815.
* **AlphaGateau Move-Graph Formulation & GATEAU Equations (Eq. 4–6, BNR Residual Blocks, Attention-Pooled Value Head, Edge Policy Head):**
  Rigaux, T., & Kashima, H. (2024). *Enhancing Chess Reinforcement Learning with Graph Representation*. Advances in Neural Information Processing Systems (NeurIPS 2024), arXiv:2410.23753. *(Conceptual and mathematical design adapted from the authors' original JAX/PGX/MCTX framework into pure PyTorch.)*
* **Supporting Methods & Frameworks:**
  * Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR 2018.
  * Danihelka, I., Guez, A., Schrittwieser, J., & Silver, D. (2022). *Policy Improvement by Planning with Gumbel*. ICLR 2022.
  * Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
  * Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
