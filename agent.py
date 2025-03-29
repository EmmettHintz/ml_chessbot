import torch
import numpy as np
import random
import os
import chess

from model import DualStreamMemoryNetwork
from utils import move_to_index, index_to_move

# Ensure we're using the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualStreamChessAgent:
    """Chess agent using the Dual-Stream Memory Network"""

    def __init__(self, model_path='chess_dsmn_model.pt', temperature=1.0, 
             conv_filters=64, hidden_size=128):
        """
        Initialize the chess agent
        """
        self.model = DualStreamMemoryNetwork(
            conv_filters=conv_filters, 
            hidden_size=hidden_size
        ).to(device)
        self.temperature = temperature
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        
        self.model.eval()

    def select_move(self, board_fen, deterministic=False):
        """
        Select a move for the given board position

        Args:
            board_fen: FEN string representing the board
            deterministic: Whether to select the best move deterministically

        Returns:
            move: The selected move in UCI format
            value: Evaluation of the position
        """
        # Get move probabilities and position evaluation
        move, probs, value = self.model.predict_move(board_fen)

        if not deterministic:
            # Sample from move distribution using temperature
            probs = probs.cpu().numpy()

            # Apply temperature
            if self.temperature != 1.0:
                probs = np.power(probs, 1.0 / self.temperature)
                probs /= np.sum(probs)

            # Sample move
            board = chess.Board(board_fen)
            legal_moves = [move.uci() for move in board.legal_moves]
            legal_moves_indices = [move_to_index(move) for move in legal_moves]

            # Filter probabilities to legal moves
            legal_probs = np.zeros_like(probs)
            legal_probs[legal_moves_indices] = probs[legal_moves_indices]
            
            # Ensure the sum is not zero before normalizing
            if np.sum(legal_probs) > 0:
                legal_probs /= np.sum(legal_probs)
            else:
                # If no legal moves have probability, use uniform distribution
                if legal_moves:
                    legal_probs[legal_moves_indices] = 1.0 / len(legal_moves_indices)

            # Sample move index
            try:
                move_idx = np.random.choice(len(probs), p=legal_probs)
                move = index_to_move(move_idx)
            except ValueError:
                # Fallback to random move if sampling fails
                move = random.choice(legal_moves) if legal_moves else None

        return move, value

    def __call__(self, observation):
        """Interface for kaggle-environments"""
        board = observation.board

        try:
            # Select move
            move, _ = self.select_move(board, deterministic=False)
            return move
        except Exception as e:
            # Fallback to random move in case of errors
            print(f"Error in agent: {e}")
            try:
                chess_board = chess.Board(board)
                legal_moves = list(chess_board.legal_moves)
                return legal_moves[random.randint(0, len(legal_moves)-1)].uci() if legal_moves else None
            except Exception as inner_e:
                print(f"Fallback error: {inner_e}")
                return None
