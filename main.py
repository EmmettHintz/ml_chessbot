import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from kaggle_environments import make
import random
import time
import os

from model import DualStreamMemoryNetwork
from agent import DualStreamChessAgent
from utils import generate_sample_games, prepare_batch, evaluate_model, train_epoch
from utils import analyze_memory_patterns, visualize_attention

# Ensure we're using the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model(
    model,
    optimizer,
    num_epochs=10,
    batch_size=32,
    seq_length=5,
    train_games=None,
    test_games=None,
    save_path="chess_dsmn_model.pt",
    scheduler=None,
):
    """
    Train and evaluate the model

    Args:
        model: The DSMN model
        optimizer: Optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size
        seq_length: Sequence length for each training example
        train_games: List of games for training
        test_games: List of games for testing
        save_path: Path to save the model

    Returns:
        history: Dictionary containing training and evaluation metrics
    """
    # Generate sample games if not provided
    if train_games is None:
        print("Generating training games...")
        train_games = generate_sample_games(num_games=100)

    if test_games is None:
        print("Generating test games...")
        test_games = generate_sample_games(num_games=20)

    # Training history
    history = {"train_loss": [], "policy_accuracy": [], "value_error": []}

    # Initial evaluation
    policy_accuracy, value_error = evaluate_model(model, test_games, seq_length)
    history["policy_accuracy"].append(policy_accuracy)
    history["value_error"].append(value_error)

    print(
        f"Initial evaluation: Policy Accuracy = {policy_accuracy:.4f}, Value Error = {value_error:.4f}"
    )

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train for one epoch
        avg_loss = train_epoch(model, optimizer, train_games, batch_size, seq_length)
        history["train_loss"].append(avg_loss)

        # Evaluate
        policy_accuracy, value_error = evaluate_model(model, test_games, seq_length)
        history["policy_accuracy"].append(policy_accuracy)
        history["value_error"].append(value_error)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, "
            f"Policy Accuracy = {policy_accuracy:.4f}, Value Error = {value_error:.4f}, "
            f"Time = {epoch_time:.2f}s"
        )

        if scheduler is not None:
            scheduler.step(value_error)  # Use validation error for scheduling

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                save_path,
            )
            print(f"Checkpoint saved to {save_path}")

    # Save final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")

    return history


def plot_training_history(history):
    """Plot training and evaluation metrics"""
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot policy accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history["policy_accuracy"])
    plt.title("Policy Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Plot value error
    plt.subplot(1, 3, 3)
    plt.plot(history["value_error"])
    plt.title("Value Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()


def play_games_vs_random(agent, num_games=10, render=True):
    """
    Play games against a random agent and evaluate performance

    Args:
        agent: The chess agent
        num_games: Number of games to play
        render: Whether to render the final position of each game

    Returns:
        results: Dictionary containing game results
    """
    env = make("chess", debug=True)
    results = {"wins": 0, "losses": 0, "draws": 0}

    for game_idx in range(num_games):
        print(f"Game {game_idx+1}/{num_games}")

        # Randomly decide which side to play
        is_white = random.choice([True, False])
        agents = (
            ["submission.py", "random"] if is_white else ["random", "submission.py"]
        )

        # Create submission.py with our agent
        with open("submission.py", "w") as f:
            f.write(
                """
                import chess
                import random
                import sys
                import os

                # Add the current directory to path
                sys.path.append(os.getcwd())

                # Global variable to store the agent
                agent = None

                def chess_bot(obs):
                    \"\"\"Chess bot using the Dual-Stream Memory Network\"\"\"
                    global agent
                    
                    # Initialize agent if not already initialized
                    if agent is None:
                        from agent import DualStreamChessAgent
                        # Use the same dimensions as during training:
                        agent = DualStreamChessAgent(conv_filters=128, hidden_size=256)
                    
                    # Select move
                    try:
                        move = agent(obs)
                        return move
                    except Exception as e:
                        # Fallback to random move
                        try:
                            board = chess.Board(obs.board)
                            legal_moves = list(board.legal_moves)
                            return legal_moves[random.randint(0, len(legal_moves)-1)].uci() if legal_moves else None
                        except Exception as inner_e:
                            # Last resort fallback
                            print(f"Error in fallback: {inner_e}")
                            return None
                """
            )

        # Play the game
        game = env.run(agents)
        steps = len(game[0]["action"]) - 1  # Number of steps taken by first player
        result = game[-1]["observation"]["reward"]

        print(f"Game completed in {steps} steps. Result: {result}")

        # Update results
        if (is_white and result == 1) or (not is_white and result == -1):
            results["wins"] += 1
        elif (is_white and result == -1) or (not is_white and result == 1):
            results["losses"] += 1
        else:
            results["draws"] += 1

        # Render final position if requested
        if render:
            env.render(mode="ipython")

    # Print results
    total_games = results["wins"] + results["losses"] + results["draws"]
    win_rate = 100 * results["wins"] / total_games if total_games > 0 else 0
    print(f"\nResults against random agent:")
    print(f"Wins: {results['wins']} ({win_rate:.1f}%)")
    print(f"Losses: {results['losses']}")
    print(f"Draws: {results['draws']}")

    return results


def main():
    # Create model
    model = DualStreamMemoryNetwork(
        conv_filters=128,  # Increased from default 64
        hidden_size=256,  # Increased from default 128
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Train model
    history = train_model(
        model,
        optimizer,
        num_epochs=20,
        batch_size=64,
        seq_length=5,
        save_path="chess_dsmn_model.pt",
        scheduler=scheduler,
    )

    # Plot training history
    plot_training_history(history)

    # Create agent with trained model
    agent = DualStreamChessAgent(
        model_path="chess_dsmn_model.pt",
        temperature=0.5,
        conv_filters=128,
        hidden_size=256,
    )

    # Play games against random agent
    play_games_vs_random(agent, num_games=5, render=True)

    # Analyze memory patterns from games (example)
    analyze_memory_patterns(model, seq_length=5, num_games=3)


if __name__ == "__main__":
    main()
