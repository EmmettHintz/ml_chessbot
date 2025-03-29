import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from matplotlib.colors import LinearSegmentedColormap
import random
from collections import defaultdict
import pandas as pd
import seaborn as sns
from datetime import datetime

from model import AlphaZeroChessNet, device
from utils import board_to_features
from mcts import MCTS


def load_training_history(history_path):
    """
    Load training history from a JSON file.
    
    Args:
        history_path: Path to the JSON file.
        
    Returns:
        Dictionary containing training history.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def plot_training_metrics(history, save_dir="visualizations"):
    """
    Plot training metrics from history.
    
    Args:
        history: Dictionary containing training history.
        save_dir: Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot policy loss
    axs[0].plot(history["iteration"], history["training_policy_loss"], 'b-')
    axs[0].set_title('Policy Loss')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    
    # Plot value loss
    axs[1].plot(history["iteration"], history["training_value_loss"], 'r-')
    axs[1].set_title('Value Loss')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)
    
    # Plot win rate
    axs[2].plot(history["iteration"], history["evaluation_win_rate"], 'g-')
    axs[2].set_title('Evaluation Win Rate')
    axs[2].set_ylabel('Win Rate')
    axs[2].set_xlabel('Iteration')
    axs[2].grid(True)
    # Add a horizontal line at win threshold (typically 0.55)
    axs[2].axhline(y=0.55, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_metrics_{timestamp}.png"))
    plt.close()


def plot_time_analysis(history, save_dir="visualizations"):
    """
    Plot time-related metrics from history.
    
    Args:
        history: Dictionary containing training history.
        save_dir: Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot time per iteration
    axs[0].bar(history["iteration"], history["time_elapsed"], color='blue', alpha=0.7)
    axs[0].set_title('Time per Iteration')
    axs[0].set_ylabel('Time (seconds)')
    axs[0].grid(True)
    
    # Plot cumulative time
    cumulative_time = np.cumsum(history["time_elapsed"])
    axs[1].plot(history["iteration"], cumulative_time / 3600, 'r-')  # Convert to hours
    axs[1].set_title('Cumulative Training Time')
    axs[1].set_ylabel('Time (hours)')
    axs[1].set_xlabel('Iteration')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"time_analysis_{timestamp}.png"))
    plt.close()


def visualize_attention_maps(model_path, positions, save_dir="visualizations"):
    """
    Visualize MCTS attention maps for given chess positions.
    
    Args:
        model_path: Path to the trained model.
        positions: List of FEN strings.
        save_dir: Directory to save visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load the model
    model = AlphaZeroChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create MCTS instance
    mcts = MCTS(model, c_puct=1.0, num_simulations=400)
    
    for i, fen in enumerate(positions):
        # Run MCTS
        action_probs, value = mcts.get_action_probs(fen, temperature=0.1)
        
        # Create a 8x8 grid for visualization
        attention_grid = np.zeros((8, 8))
        
        # Map each move's probability to the target square
        for move_idx, prob in enumerate(action_probs):
            if prob > 0:
                # Convert move index to target square coordinates
                try:
                    move = mcts.index_to_move(move_idx)
                    to_file = ord(move[2]) - ord('a')
                    to_rank = 8 - int(move[3])
                    
                    # Add probability to that square
                    attention_grid[to_rank, to_file] += prob
                except:
                    continue
        
        # Create a chessboard-like background
        chessboard = np.zeros((8, 8, 3))
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    chessboard[r, c] = [0.9, 0.9, 0.8]  # Light square
                else:
                    chessboard[r, c] = [0.5, 0.6, 0.4]  # Dark square
        
        # Create a custom colormap (transparent to red)
        colors = [(0, 0, 0, 0), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
        
        # Create the plot
        plt.figure(figsize=(10, 10))
        
        # Plot the chessboard background
        plt.imshow(chessboard)
        
        # Overlay the attention heatmap
        plt.imshow(attention_grid, cmap=cmap, alpha=0.7, vmin=0, vmax=attention_grid.max())
        
        # Add position evaluation
        plt.title(f"Position Evaluation: {value:.2f}", fontsize=14)
        
        # Add a colorbar
        cbar = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Move Probability', fontsize=12)
        
        # Set board coordinates
        plt.xticks(np.arange(8), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        plt.yticks(np.arange(8), ['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f"attention_map_{i}_{timestamp}.png"))
        plt.close()


def compare_model_differences(model1_path, model2_path, save_dir="visualizations"):
    """
    Compare and visualize differences between two model checkpoints.
    
    Args:
        model1_path: Path to the first model.
        model2_path: Path to the second model.
        save_dir: Directory to save visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load models
    model1 = AlphaZeroChessNet().to(device)
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    
    model2 = AlphaZeroChessNet().to(device)
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    
    # Compare parameters
    differences = {}
    total_params = 0
    different_params = 0
    
    for name, param1 in model1.named_parameters():
        param2 = dict(model2.named_parameters())[name]
        total_params += param1.numel()
        
        # Calculate absolute difference
        diff = torch.abs(param1 - param2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        different_params += torch.sum(diff > 0.01).item()  # Count params with significant difference
        
        differences[name] = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "shape": list(param1.shape),
            "num_params": param1.numel()
        }
    
    # Create a DataFrame for visualization
    diff_data = []
    for name, data in differences.items():
        diff_data.append({
            "layer": name,
            "max_diff": data["max_diff"],
            "mean_diff": data["mean_diff"],
            "num_params": data["num_params"]
        })
    
    df = pd.DataFrame(diff_data)
    
    # Sort by maximum difference
    df = df.sort_values(by="max_diff", ascending=False)
    
    # Calculate the percentage of parameters that changed significantly
    percent_changed = (different_params / total_params) * 100
    
    # Create a bar plot of max differences by layer
    plt.figure(figsize=(14, 10))
    sns.barplot(x="max_diff", y="layer", data=df.head(20), palette="coolwarm")
    plt.title(f"Top 20 Layers with Maximum Parameter Differences\n{percent_changed:.2f}% of parameters changed significantly", fontsize=14)
    plt.xlabel("Maximum Absolute Difference", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"model_diff_max_{timestamp}.png"))
    plt.close()
    
    # Create a bar plot of mean differences by layer
    plt.figure(figsize=(14, 10))
    sns.barplot(x="mean_diff", y="layer", data=df.sort_values(by="mean_diff", ascending=False).head(20), palette="coolwarm")
    plt.title("Top 20 Layers with Mean Parameter Differences", fontsize=14)
    plt.xlabel("Mean Absolute Difference", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"model_diff_mean_{timestamp}.png"))
    plt.close()
    
    # Save the full data to CSV
    df.to_csv(os.path.join(save_dir, f"model_diff_data_{timestamp}.csv"), index=False)


def visualize_game_statistics(evaluation_results_dir, save_dir="visualizations"):
    """
    Analyze and visualize statistics from evaluation games.
    
    Args:
        evaluation_results_dir: Directory containing evaluation result JSON files.
        save_dir: Directory to save visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all JSON files in the evaluation directory
    json_files = [os.path.join(evaluation_results_dir, f) for f in os.listdir(evaluation_results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {evaluation_results_dir}")
        return
    
    # Collect statistics from all games
    game_stats = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            eval_data = json.load(f)
            
            for game in eval_data.get("games_data", []):
                game_stats.append({
                    "game_id": game["game_id"],
                    "model1_plays_white": game["model1_plays_white"],
                    "result": game["result"],  # 1: model1 wins, -1: model2 wins, 0: draw
                    "move_count": game["move_count"],
                    "game_time": game["game_time"]
                })
    
    if not game_stats:
        print("No game data found in the evaluation files")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(game_stats)
    
    # Game outcome statistics
    outcomes = df["result"].value_counts().reset_index()
    outcomes.columns = ["outcome", "count"]
    outcomes["outcome"] = outcomes["outcome"].map({1: "Model1 Wins", -1: "Model2 Wins", 0: "Draw"})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="outcome", y="count", data=outcomes, palette="viridis")
    plt.title("Game Outcomes", fontsize=14)
    plt.ylabel("Number of Games", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"game_outcomes_{timestamp}.png"))
    plt.close()
    
    # Game length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["move_count"], bins=20, kde=True)
    plt.title("Distribution of Game Lengths", fontsize=14)
    plt.xlabel("Number of Moves", fontsize=12)
    plt.ylabel("Number of Games", fontsize=12)
    plt.axvline(x=df["move_count"].mean(), color='r', linestyle='--', label=f"Mean: {df['move_count'].mean():.1f} moves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"game_lengths_{timestamp}.png"))
    plt.close()
    
    # Outcome by side (white/black)
    side_outcome = pd.crosstab(
        df["model1_plays_white"], 
        df["result"].map({1: "Model1 Wins", -1: "Model2 Wins", 0: "Draw"})
    )
    
    plt.figure(figsize=(12, 6))
    side_outcome.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Game Outcomes by Side", fontsize=14)
    plt.xlabel("Model1 Plays White", fontsize=12)
    plt.ylabel("Number of Games", fontsize=12)
    plt.xticks([0, 1], ["No (Black)", "Yes (White)"], rotation=0)
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"outcomes_by_side_{timestamp}.png"))
    plt.close()
    
    # Win rate by side
    win_rate_by_side = df.groupby("model1_plays_white")["result"].apply(
        lambda x: (x == 1).sum() / len(x)
    ).reset_index()
    win_rate_by_side.columns = ["model1_plays_white", "win_rate"]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model1_plays_white", y="win_rate", data=win_rate_by_side)
    plt.title("Model1 Win Rate by Side", fontsize=14)
    plt.xlabel("Model1 Plays White", fontsize=12)
    plt.ylabel("Win Rate", fontsize=12)
    plt.xticks([0, 1], ["No (Black)", "Yes (White)"])
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"win_rate_by_side_{timestamp}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training Visualization Tools")
    
    # Input files and directories
    parser.add_argument("--history", type=str, default="models/training_history.json", help="Path to training history JSON file")
    parser.add_argument("--eval_dir", type=str, default="evaluation_results", help="Directory with evaluation results")
    parser.add_argument("--model1", type=str, default=None, help="Path to first model for comparison")
    parser.add_argument("--model2", type=str, default=None, help="Path to second model for comparison")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save visualizations")
    
    # Types of visualizations to create
    parser.add_argument("--training_metrics", action="store_true", help="Plot training metrics")
    parser.add_argument("--time_analysis", action="store_true", help="Plot time-related metrics")
    parser.add_argument("--attention_maps", action="store_true", help="Visualize attention maps for positions")
    parser.add_argument("--model_diff", action="store_true", help="Compare two model checkpoints")
    parser.add_argument("--game_stats", action="store_true", help="Analyze game statistics")
    parser.add_argument("--all", action="store_true", help="Generate all available visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine which visualizations to create
    if args.all:
        args.training_metrics = True
        args.time_analysis = True
        args.attention_maps = True
        args.model_diff = True
        args.game_stats = True
    
    # Load training history if needed
    if args.training_metrics or args.time_analysis:
        try:
            history = load_training_history(args.history)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            if not (args.attention_maps or args.model_diff or args.game_stats):
                return
    
    # Generate requested visualizations
    if args.training_metrics:
        print("Generating training metrics plots...")
        plot_training_metrics(history, args.save_dir)
    
    if args.time_analysis:
        print("Generating time analysis plots...")
        plot_time_analysis(history, args.save_dir)
    
    if args.attention_maps:
        if args.model1:
            print("Generating attention maps...")
            positions = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Common opening
                "r3k2r/ppp2ppp/2n1b3/2b1p3/2B1P3/2N5/PPP2PPP/R3K2R w KQkq - 0 10",  # Middlegame with castling options
                "8/3k4/8/8/8/8/3K1R2/8 w - - 0 1"  # Basic endgame
            ]
            visualize_attention_maps(args.model1, positions, args.save_dir)
        else:
            print("Warning: No model provided for attention map visualization")
    
    if args.model_diff:
        if args.model1 and args.model2:
            print("Comparing model differences...")
            compare_model_differences(args.model1, args.model2, args.save_dir)
        else:
            print("Warning: Need both --model1 and --model2 for model comparison")
    
    if args.game_stats:
        print("Analyzing game statistics...")
        visualize_game_statistics(args.eval_dir, args.save_dir)
    
    print(f"All visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    main() 