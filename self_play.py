import os
import numpy as np
import torch
import time
import random
from collections import deque
import multiprocessing
import argparse
from datetime import datetime
import json
import chess # Use python-chess

from model import AlphaZeroChessNet, device
from mcts import MCTS
from utils import board_to_features, move_to_index, index_to_move

class SelfPlay:
    """
    Manages the self-play process to generate training data using python-chess.
    """
    def __init__(
        self,
        model,
        num_games=100,
        num_simulations=100,
        c_puct=1.0,
        temperature_threshold=10,
        dir_alpha=0.3,
        exploration_fraction=0.25,
        max_moves=512,
        checkpoint_frequency=5,
        data_dir="training_data",
    ):
        """
        Initialize the self-play manager.
        Args are the same, but implementation uses python-chess.
        """
        self.model = model
        self.num_games = num_games
        # MCTS instance should already be using python-chess
        self.mcts = MCTS(self.model, c_puct=c_puct, num_simulations=num_simulations)
        self.temperature_threshold = temperature_threshold
        self.max_moves = max_moves
        self.checkpoint_frequency = checkpoint_frequency
        self.data_dir = data_dir

        # Dirichlet noise parameters (can be used in MCTS if desired, currently not implemented in MCTS provided)
        self.dir_alpha = dir_alpha
        self.exploration_fraction = exploration_fraction

        os.makedirs(self.data_dir, exist_ok=True)

    def play_game(self, game_id=0, verbose=False):
        """
        Play a full game of chess using MCTS for move selection (python-chess based).
        
        Args:
            game_id: Identifier for the game (for logging).
            verbose: Whether to print game status during play.
            
        Returns:
            A list of tuples (state_fen, mcts_policy, result_from_perspective).
        """
        board = chess.Board() # Starting position
        states = []
        mcts_policies = []
        current_player_perspective = [] # Stores 1 if White to move, -1 if Black to move
        
        move_count = 0
        start_time = time.time()
        
        while not board.is_game_over(claim_draw=True) and move_count < self.max_moves:
            board_fen = board.fen()
            states.append(board_fen)
            player = 1 if board.turn == chess.WHITE else -1
            current_player_perspective.append(player)
            
            # Set temperature
            temp = 1.0 if move_count < self.temperature_threshold else 0.0
            
            # Get policy from MCTS
            # Add Dirichlet noise if temp > 0? (Optional, often done inside MCTS root node)
            action_probs, _ = self.mcts.get_action_probs(board_fen, temperature=temp)
            mcts_policies.append(action_probs)
            
            # Select move based on the policy
            # Note: action_probs should already be masked for legal moves by MCTS
            valid_move_indices = np.where(action_probs > 0)[0]
            if len(valid_move_indices) == 0:
                 print(f"Error: MCTS returned no valid moves for FEN: {board_fen}. Game {game_id} aborted.")
                 # This should ideally not happen if MCTS is correct. Maybe return partial data?
                 break

            # Sample or choose best move
            if temp > 0:
                 # Sample proportionally to visit counts (represented by action_probs)
                 move_idx = np.random.choice(valid_move_indices, p=action_probs[valid_move_indices])
            else:
                 # Choose the move with the highest visit count
                 move_idx = valid_move_indices[np.argmax(action_probs[valid_move_indices])]
            
            # Convert index back to move object
            move_uci = index_to_move(move_idx)
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    # This indicates a serious issue in MCTS policy or move indexing
                    print(f"CRITICAL Error: MCTS suggested illegal move {move_uci} for FEN {board_fen}. Aborting game {game_id}.")
                    break # Abort this game
            except ValueError:
                print(f"CRITICAL Error: Invalid move UCI '{move_uci}' (index {move_idx}) generated. Aborting game {game_id}.")
                break # Abort this game

            if verbose and move_count % 5 == 0:
                print(f"Game {game_id}, Move {move_count}: {move.uci()}, FEN: {board_fen}")
            
            # Apply the move
            board.push(move)
            move_count += 1
        
        # Game over - determine the result
        final_result = 0 # Default to draw
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE:
                final_result = 1
            elif outcome.winner == chess.BLACK:
                final_result = -1
            else: # Draw
                final_result = 0
        elif move_count >= self.max_moves:
             print(f"Game {game_id} force-drawn due to max moves ({self.max_moves}).")
             final_result = 0 # Draw due to move limit
        
        # Convert result to perspective of player at each state
        # result_z = final_result if player_at_state == white else -final_result
        results_z = [final_result * perspective for perspective in current_player_perspective]
        
        training_examples = [(s, p, r) for s, p, r in zip(states, mcts_policies, results_z)]
        
        game_time = time.time() - start_time
        result_str = "Draw" if final_result == 0 else ("White Win" if final_result == 1 else "Black Win")
        if verbose:
            print(f"Game {game_id} complete in {move_count} moves ({game_time:.1f}s). Result: {result_str}. Final FEN: {board.fen()}")
        
        return training_examples

    def generate_data(self, verbose=True):
        """
        Generate self-play data by playing multiple games.
        """
        all_examples = []
        for game_id in range(self.num_games):
            if verbose:
                print(f"\n--- Starting game {game_id + 1}/{self.num_games} ---")
            
            examples = self.play_game(game_id, verbose=verbose)
            if examples: # Only add if game wasn't aborted early due to errors
                all_examples.extend(examples)
            
            if (game_id + 1) % self.checkpoint_frequency == 0:
                self.save_examples(all_examples[-(len(examples)*self.checkpoint_frequency):], is_checkpoint=True, checkpoint_id=game_id + 1)
        
        # Save final full dataset (optional, might be redundant with checkpoints)
        # self.save_examples(all_examples, is_checkpoint=False)
        
        if verbose:
            print(f"\n--- Self-play generation complete. Generated {len(all_examples)} examples from ~{self.num_games} games. ---")
        
        return all_examples

    def save_examples(self, examples, is_checkpoint=False, checkpoint_id=None):
        """
        Save the training examples to disk.
        
        Args:
            examples: List of (state, mcts_policy, result) tuples.
            is_checkpoint: Whether this is a periodic checkpoint save.
            checkpoint_id: Identifier for the checkpoint (e.g., game number).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_checkpoint:
            filename = f"self_play_data_checkpoint_{checkpoint_id}_{timestamp}.npz"
        else:
            filename = f"self_play_data_full_{timestamp}.npz"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Extract data components
        states = [ex[0] for ex in examples]  # FEN strings
        policies = [ex[1] for ex in examples]  # MCTS policies (numpy arrays)
        results = [ex[2] for ex in examples]  # Game results (scalars)
        
        # Save as NumPy compressed array
        try:
            np.savez_compressed(
                filepath,
                states=np.array(states, dtype=object), # FEN strings
                policies=np.array(policies, dtype=np.float32),
                results=np.array(results, dtype=np.float32)
            )
            print(f"Saved {len(examples)} training examples to {filepath}")
        except Exception as e:
            print(f"Error saving examples to {filepath}: {e}")


# Function to preprocess data for training
def preprocess_example(fen, policy, result):
    """
    Convert a training example to the format needed for the neural network.
    
    Args:
        fen: FEN string representing the board state.
        policy: MCTS policy (numpy array).
        result: Game result from this position's player's perspective.
        
    Returns:
        x: Input features for the neural network.
        policy: MCTS policy (numpy array).
        value: Game result as a scalar.
    """
    x = board_to_features(fen)
    return x, policy, result


# Entry point for running self-play
def main(args):
    # Initialize the model
    print(f"Initializing model...")
    model = AlphaZeroChessNet(
        input_channels=18,  # Adjust based on your feature representation
        num_filters=256,
        num_res_blocks=19,
        policy_output_size=1968 # Make sure this matches utils.POLICY_OUTPUT_SIZE
    ).to(device)
    
    # Load weights if available
    weights_path = args.weights
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        try:
            # Load model state dict
            checkpoint = torch.load(weights_path, map_location=device)
            # Adjust loading based on how weights were saved (e.g., if saved as checkpoint dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 model.load_state_dict(checkpoint)
        except Exception as e:
             print(f"Error loading weights from {weights_path}: {e}. Starting with random weights.")
    else:
        print("Using randomly initialized weights")
    
    model.eval()
    
    # Create self-play manager
    self_play = SelfPlay(
        model=model,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature_threshold=args.temp_threshold,
        dir_alpha=args.dir_alpha,
        exploration_fraction=args.exploration_fraction,
        max_moves=args.max_moves,
        checkpoint_frequency=args.checkpoint_freq,
        data_dir=args.data_dir
    )
    
    # Generate self-play data
    print(f"Starting self-play for {args.num_games} games with {args.num_simulations} MCTS simulations per move...")
    examples = self_play.generate_data(verbose=True)
    
    print(f"Self-play complete. Generated {len(examples)} training examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Chess Self-Play")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file")
    parser.add_argument("--num_games", type=int, default=50, help="Number of self-play games to generate")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of MCTS simulations per move")
    parser.add_argument("--c_puct", type=float, default=1.0, help="PUCT exploration constant")
    parser.add_argument("--temp_threshold", type=int, default=15, help="Move threshold for temperature=0")
    parser.add_argument("--dir_alpha", type=float, default=0.3, help="Dirichlet noise alpha parameter")
    parser.add_argument("--exploration_fraction", type=float, default=0.25, help="Fraction of root prior to replace with noise")
    parser.add_argument("--max_moves", type=int, default=512, help="Maximum moves per game before force draw")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Save data after this many games") # Increased default
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory to save training data")
    
    args = parser.parse_args()
    main(args) 