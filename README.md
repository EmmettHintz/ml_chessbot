# AlphaZero Chess Bot

This project implements an AlphaZero-style algorithm for training a chess bot using PyTorch and the `python-chess` library. The goal is to train a neural network through self-play and Monte Carlo Tree Search (MCTS) to predict optimal moves and evaluate board positions.

## Features

-   **AlphaZero Algorithm:** Implements the core loop of self-play data generation and neural network training.
-   **Monte Carlo Tree Search (MCTS):** Uses MCTS guided by the neural network's policy and value predictions to select moves during self-play.
-   **PyTorch Neural Network:** A convolutional neural network (`AlphaZeroChessNet`) serves as the function approximator, outputting move probabilities (policy) and position evaluation (value).
-   **Self-Play:** Generates training data by having the current best model play games against itself.
-   **`python-chess` Integration:** Leverages the `python-chess` library for board representation, move generation, and legality checking.
-   **Custom Move Representation:** Uses a defined mapping for encoding/decoding chess moves for the policy head, handling the large action space of chess.

## Installation

Create a Python 3.10 virtual environment:

```bash
python3.10 -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies (ensure requirements.txt is up-to-date)
pip install -r requirements.txt
```

## Project Structure

-   `alphazero_main.py`: Main script orchestrating the AlphaZero training pipeline (self-play and training iterations).
-   `model.py`: Contains the `AlphaZeroChessNet` neural network architecture using PyTorch.
-   `mcts.py`: Implements the Monte Carlo Tree Search algorithm.
-   `self_play.py`: Handles the process of generating game data through self-play using MCTS.
-   `utils.py`: Includes helper functions, notably for board-to-tensor conversion, move encoding/decoding, and generating the legal move mask based on the network's policy output.
-   `requirements.txt`: Project dependencies.

## Usage

### Training the Model

To run the AlphaZero training pipeline:

```bash
python alphazero_main.py --iterations <num_iterations> --self-play-games <games_per_iteration> --train-epochs <epochs_per_iteration>
```

-   `--iterations`: Total number of AlphaZero iterations (self-play + training cycles) to run.
-   `--self-play-games`: Number of games to generate via self-play in each iteration.
-   `--train-epochs`: Number of training epochs to perform on the generated data in each iteration.

The script will automatically:
1.  Load the latest saved model weights if they exist, otherwise start with a random network.
2.  Run the self-play phase using the current network and MCTS to generate game data.
3.  Train the network on the newly generated data.
4.  Save the updated model weights.
5.  Repeat for the specified number of iterations.

## How it Works (AlphaZero Core Loop)

1.  **Self-Play:** The current best neural network plays games against itself. For each move:
    *   **MCTS:** An MCTS simulation is run from the current board state. The search is guided by the network's policy (suggesting promising moves) and value (evaluating resulting positions).
    *   **Move Selection:** A move is chosen based on the MCTS visit counts (usually probabilistically, weighted by visits, especially early in the game).
    *   **Data Collection:** The board state, the final MCTS policy (visit counts), and the eventual game outcome (win/loss/draw) are stored as training examples.
2.  **Training:** The neural network is trained on batches of data collected from multiple self-play games:
    *   **Input:** A board state.
    *   **Targets:**
        *   **Policy Head:** Trained to match the MCTS visit count distribution (π) from the self-play phase for that state.
        *   **Value Head:** Trained to predict the actual outcome (z) of the game from that state (+1 for win, -1 for loss, 0 for draw).
3.  **Iteration:** The newly trained network becomes the "current best" network for the next iteration of self-play, leading to stronger play and higher-quality training data over time.

## Implementation Notes & Potential Novelty

Compared to a textbook AlphaZero implementation or other engines, this project focuses on:

-   **Practical `python-chess` Integration:** Demonstrates the integration of `python-chess` for core game logic within the AlphaZero framework.
-   **Move Space Handling:** Explicitly addresses the large action space of chess (thousands of possible moves) by using a fixed-size policy output (1968 moves in this case). It includes logic (`get_legal_moves_mask`) to correctly handle situations where the true legal moves are not all represented in the network's output head, falling back to a uniform probability distribution over the actual legal moves when necessary. This is a practical compromise often required.
-   **Iterative Development:** The history of this project reflects a migration from previous architectures (like DSMN) and debugging steps involved in adapting standard algorithms to the specific libraries and constraints used.
