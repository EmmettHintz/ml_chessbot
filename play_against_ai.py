import os
import argparse
import torch
import random
import time
import numpy as np
import re

from model import AlphaZeroChessNet, device
from mcts import MCTS

# Use python-chess for the board representation
import chess
import chess.svg

# Optional: Try to import rendering libraries for better visualization
try:
    import cairosvg
    import PIL.Image
    import io
    HAS_CHESS_RENDER = True
except ImportError:
    HAS_CHESS_RENDER = False


def print_board(board, last_move=None, highlight=True):
    """
    Print a chess board representation in the terminal.
    
    Args:
        board: python-chess Board object
        last_move: Last move in UCI format or None.
        highlight: Whether to highlight the last move.
    """
    # Map for translating pieces
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        ' ': ' '
    }
    
    # Parse last move to highlight
    highlight_squares = set()
    if highlight and last_move and len(last_move) >= 4:
        from_square = chess.parse_square(last_move[:2])
        to_square = chess.parse_square(last_move[2:4])
        highlight_squares.add(from_square)
        highlight_squares.add(to_square)
    
    # Print the board
    print("  +------------------------+")
    for rank in range(7, -1, -1):
        print(f"{rank+1} |", end=" ")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = ' '
            if piece:
                symbol = piece_symbols.get(piece.symbol(), piece.symbol())
            
            # Add highlighting (if available)
            if square in highlight_squares and highlight:
                print(f"\033[42m{symbol}\033[0m", end=" ")  # Green background
            else:
                # Alternate square colors
                if (rank + file) % 2 == 0:
                    print(f"\033[47m\033[30m{symbol}\033[0m", end=" ")  # White background
                else:
                    print(f"\033[40m\033[37m{symbol}\033[0m", end=" ")  # Black background
        print("|")
    print("  +------------------------+")
    print("    a b c d e f g h")


def save_board_image(board, move=None, output_path='board.png'):
    """
    Save a chess board image using python-chess and cairosvg.
    Requires python-chess and cairosvg to be installed.
    
    Args:
        board: python-chess Board object
        move: Last move in UCI format or None.
        output_path: Path to save the image.
    """
    if not HAS_CHESS_RENDER:
        print("Skipping image rendering - required packages not installed.")
        print("To enable, install: pip install python-chess cairosvg pillow")
        return
    
    last_move = None
    
    if move and len(move) >= 4:
        try:
            from_square = chess.parse_square(move[0:2])
            to_square = chess.parse_square(move[2:4])
            last_move = chess.Move(from_square, to_square)
            
            # Handle promotion if present
            if len(move) > 4:
                promotion_piece = move[4].lower()
                if promotion_piece == 'q':
                    last_move.promotion = chess.QUEEN
                elif promotion_piece == 'r':
                    last_move.promotion = chess.ROOK
                elif promotion_piece == 'b':
                    last_move.promotion = chess.BISHOP
                elif promotion_piece == 'n':
                    last_move.promotion = chess.KNIGHT
        except ValueError:
            # Invalid move format
            pass
    
    # Generate SVG
    svg_content = chess.svg.board(
        board=board,
        lastmove=last_move,
        size=400
    )
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(png_data)


class ChessGame:
    """
    A class to manage the chess game between human and AI.
    """
    def __init__(self, model_path, num_simulations=800, temperature=0.1):
        """
        Initialize the chess game.
        
        Args:
            model_path: Path to the trained model.
            num_simulations: Number of MCTS simulations per move.
            temperature: Temperature for move selection.
        """
        # Load model
        self.model = AlphaZeroChessNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Create MCTS
        self.mcts = MCTS(self.model, c_puct=1.0, num_simulations=num_simulations)
        
        # Game state
        self.board = chess.Board()
        self.current_fen = self.board.fen()
        self.move_history = []
        self.last_move = None
        self.temperature = temperature
        
        # Display settings
        self.save_images = HAS_CHESS_RENDER
        self.image_path = "current_board.png"
    
    def get_player_move(self):
        """
        Get a move from the human player.
        
        Returns:
            A valid move in UCI format.
        """
        if self.board.legal_moves.count() == 0:
            return None
        
        while True:
            try:
                move_input = input("\nYour move (e.g., e2e4): ").strip().lower()
                
                # Handle special commands
                if move_input == "quit" or move_input == "exit":
                    return "quit"
                elif move_input == "help":
                    print("\nCommands:")
                    print("  <move>   - Make a move (e.g., e2e4)")
                    print("  legal    - Show legal moves")
                    print("  fen      - Show current position in FEN")
                    print("  quit     - Exit the game")
                    print("  help     - Show this help message")
                    continue
                elif move_input == "legal":
                    legal_moves = [move.uci() for move in self.board.legal_moves]
                    print("Legal moves:", ", ".join(legal_moves))
                    continue
                elif move_input == "fen":
                    print("Current FEN:", self.current_fen)
                    continue
                
                # Validate move format
                if not re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', move_input):
                    print("Invalid format. Use standard UCI notation (e.g., e2e4, a7a8q)")
                    continue
                
                # Check if move is legal
                try:
                    move = chess.Move.from_uci(move_input)
                    if move in self.board.legal_moves:
                        return move_input
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid move. Try again.")
            except Exception as e:
                print(f"Error: {e}")
    
    def get_ai_move(self):
        """
        Get a move from the AI.
        
        Returns:
            A move in UCI format.
        """
        if self.board.legal_moves.count() == 0:
            return None
        
        print("\nAI is thinking...")
        start_time = time.time()
        
        # Get probabilities from MCTS
        probabilities, _ = self.mcts.get_action_probs(self.current_fen, self.temperature)
        
        # Choose a move based on the probabilities
        move_idx = np.random.choice(len(probabilities), p=probabilities)
        move_uci = None
        
        # Convert move_idx to UCI
        from utils import index_to_move
        move_uci = index_to_move(move_idx)
        
        # Validate move with python-chess
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                # If the move is not legal, choose a random legal move
                # This is a fallback in case the MCTS produces an invalid move
                legal_moves = list(self.board.legal_moves)
                move = random.choice(legal_moves)
                move_uci = move.uci()
                print(f"Warning: MCTS selected illegal move, using fallback: {move_uci}")
        except ValueError:
            # If the move is not a valid UCI string, choose a random legal move
            legal_moves = list(self.board.legal_moves)
            move = random.choice(legal_moves)
            move_uci = move.uci()
            print(f"Warning: MCTS selected invalid move, using fallback: {move_uci}")
        
        time_taken = time.time() - start_time
        print(f"AI move: {move_uci} (took {time_taken:.1f} seconds)")
        
        return move_uci
    
    def apply_move(self, move_uci):
        """
        Apply a move to the game state.
        
        Args:
            move_uci: Move in UCI format.
            
        Returns:
            True if the move was applied successfully, False otherwise.
        """
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move_uci
                self.move_history.append(move_uci)
                self.current_fen = self.board.fen()
                return True
            else:
                print(f"Error: Illegal move: {move_uci}")
                return False
        except ValueError:
            print(f"Error: Invalid move format: {move_uci}")
            return False
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            True if the game is over, False otherwise.
        """
        return self.board.is_game_over()
    
    def get_game_result(self):
        """
        Get the result of the game.
        
        Returns:
            "1-0" for white win, "0-1" for black win, "1/2-1/2" for draw.
        """
        if not self.is_game_over():
            return None
        
        outcome = self.board.outcome()
        if outcome is None:  # Shouldn't happen if is_game_over() is True
            return "1/2-1/2"
        
        if outcome.winner == chess.WHITE:
            return "1-0"
        elif outcome.winner == chess.BLACK:
            return "0-1"
        else:
            return "1/2-1/2"
    
    def display_board(self):
        """
        Display the current board state.
        """
        print("\nCurrent position:")
        print_board(self.board, self.last_move)
        
        if self.save_images:
            save_board_image(self.board, self.last_move, self.image_path)
            print(f"Board image saved to {self.image_path}")
    
    def play_game(self, player_color='w'):
        """
        Play a full game between the human player and the AI.
        
        Args:
            player_color: The color the human player will play ('w' or 'b').
            
        Returns:
            The result of the game.
        """
        # Convert player_color to boolean for compatibility with python-chess
        player_is_white = player_color.lower() == 'w'
        
        # Display initial board
        self.display_board()
        
        # Main game loop
        while not self.is_game_over():
            # Determine current player
            is_player_turn = (player_is_white and self.board.turn == chess.WHITE) or \
                            (not player_is_white and self.board.turn == chess.BLACK)
            
            if is_player_turn:
                # Human player's turn
                print("\nYour turn.")
                move = self.get_player_move()
                
                if move == "quit":
                    print("\nGame aborted.")
                    return None
                
                success = self.apply_move(move)
                if not success:
                    continue  # Try again if move failed
            else:
                # AI's turn
                move = self.get_ai_move()
                if move is None:
                    print("\nError: AI couldn't find a move.")
                    break
                
                self.apply_move(move)
            
            # Display board after move
            self.display_board()
            
            # Short delay for better user experience
            time.sleep(0.5)
        
        # Game over
        result = self.get_game_result()
        print("\nGame over!")
        
        if result == "1-0":
            print("White wins!")
        elif result == "0-1":
            print("Black wins!")
        else:
            print("It's a draw!")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Play chess against AlphaZero AI')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--simulations', type=int, default=800, help='Number of MCTS simulations')
    parser.add_argument('--color', type=str, default='w', choices=['w', 'b'], help='Your color (w or b)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for move selection')
    args = parser.parse_args()
    
    game = ChessGame(
        model_path=args.model,
        num_simulations=args.simulations,
        temperature=args.temperature
    )
    
    print("\nWelcome to AlphaZero Chess!")
    print(f"You are playing as {'White' if args.color == 'w' else 'Black'}")
    print("Type 'help' for commands. Let's begin!\n")
    
    game.play_game(player_color=args.color)
    
    print("\nThanks for playing!")


if __name__ == "__main__":
    main() 