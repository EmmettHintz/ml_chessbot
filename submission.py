import chess
import random
import sys
import os

# Add the current directory to path
sys.path.append(os.getcwd())

# Global variable to store the agent
agent = None

def chess_bot(obs):
    """Chess bot using the Dual-Stream Memory Network"""
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
            # Last-resort fallback
            print(f"Error in fallback: {inner_e}")
            return None
                