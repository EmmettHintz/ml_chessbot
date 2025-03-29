import os
import json
import random
import re
import logging
import numpy as np
import shutil
import requests
import zipfile
import chess
import chess.polyglot
import chess.syzygy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class OpeningBook:
    """
    Chess opening book for improved play in the opening phase.
    Uses Python-chess's polyglot format support.
    """
    def __init__(self, book_path=None, download_if_missing=True):
        """
        Initialize the opening book.
        
        Args:
            book_path: Path to the polyglot opening book (.bin file)
            download_if_missing: Whether to download a book if not found
        """
        self.book_path = book_path or "data/books/gm2600.bin"
        self.reader = None
        self.enabled = True
        
        # Try to load the book
        if not self.load_book() and download_if_missing:
            self.download_default_book()
            self.load_book()
    
    def load_book(self):
        """
        Load the opening book from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if os.path.exists(self.book_path):
                self.reader = chess.polyglot.open_reader(self.book_path)
                logger.info(f"Opening book loaded from {self.book_path}")
                return True
            else:
                logger.warning(f"Opening book not found at {self.book_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading opening book: {e}")
            self.enabled = False
            return False
    
    def download_default_book(self):
        """
        Download a default opening book.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.book_path), exist_ok=True)
        
        # URL for a commonly used opening book (gm2600.bin)
        url = "https://github.com/maksimKorzh/chess_opening_books/raw/master/gm2600.bin"
        
        try:
            logger.info(f"Downloading opening book from {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(self.book_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                logger.info(f"Opening book downloaded to {self.book_path}")
            else:
                logger.error(f"Failed to download opening book: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading opening book: {e}")
    
    def get_move(self, board, weight_by_probability=True):
        """
        Get a move from the opening book.
        
        Args:
            board: A chess.Board object.
            weight_by_probability: Whether to weight move selection by occurrence count.
            
        Returns:
            A chess.Move object, or None if no move is found.
        """
        if not self.enabled or not self.reader:
            return None
        
        try:
            # Get all matching moves from the book
            entries = list(self.reader.find_all(board))
            if not entries:
                return None
            
            if weight_by_probability:
                # Weight by frequency/quality (weight)
                total_weight = sum(entry.weight for entry in entries)
                
                if total_weight == 0:
                    # If all weights are 0, select randomly
                    return random.choice(entries).move
                
                # Select a move probabilistically based on its weight
                choice = random.uniform(0, total_weight)
                current_weight = 0
                
                for entry in entries:
                    current_weight += entry.weight
                    if current_weight >= choice:
                        return entry.move
                
                # Fallback to the highest weighted move if something goes wrong
                return max(entries, key=lambda x: x.weight).move
            else:
                # Simple random selection
                return random.choice(entries).move
        except Exception as e:
            logger.error(f"Error retrieving move from opening book: {e}")
            return None
    
    def close(self):
        """Close the opening book reader."""
        if self.reader:
            self.reader.close()


class EndgameTablebase:
    """
    Endgame tablebase support for perfect play in supported endgames.
    Uses Python-chess's syzygy format support.
    """
    def __init__(self, tablebase_dir=None, download_if_missing=True, max_pieces=5):
        """
        Initialize the endgame tablebase.
        
        Args:
            tablebase_dir: Directory containing Syzygy tablebase files (.rtbw and .rtbz)
            download_if_missing: Whether to download tablebases if not found
            max_pieces: Maximum number of pieces for which to download tablebases
        """
        self.tablebase_dir = tablebase_dir or "data/tablebases/syzygy"
        self.max_pieces = max_pieces
        self.enabled = True
        self.tablebase = None
        
        # Try to load the tablebases
        if not self.load_tablebases() and download_if_missing:
            self.download_tablebases(max_pieces)
            self.load_tablebases()
    
    def load_tablebases(self):
        """
        Load the tablebases from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if os.path.exists(self.tablebase_dir) and os.listdir(self.tablebase_dir):
                self.tablebase = chess.syzygy.open_tablebase(self.tablebase_dir)
                pieces = self.detect_available_pieces()
                logger.info(f"Loaded Syzygy tablebases for up to {pieces} pieces from {self.tablebase_dir}")
                return True
            else:
                logger.warning(f"No tablebase files found in {self.tablebase_dir}")
                return False
        except Exception as e:
            logger.error(f"Error loading tablebases: {e}")
            self.enabled = False
            return False
    
    def detect_available_pieces(self):
        """
        Detect the maximum number of pieces supported by the loaded tablebases.
        
        Returns:
            Maximum number of pieces supported.
        """
        rtbw_files = [f for f in os.listdir(self.tablebase_dir) if f.endswith('.rtbw')]
        if not rtbw_files:
            return 0
        
        # Parse piece count from filenames like "KPvK.rtbw" (3 pieces)
        max_pieces = 0
        for file in rtbw_files:
            # Count pieces in filename (K, Q, R, B, N, P)
            piece_count = len(re.findall('[KQRBNP]', file, re.IGNORECASE))
            max_pieces = max(max_pieces, piece_count)
        
        return max_pieces
    
    def download_tablebases(self, max_pieces=5):
        """
        Download Syzygy tablebases.
        
        Args:
            max_pieces: Maximum number of pieces for which to download tablebases.
        """
        # Create directory if it doesn't exist
        os.makedirs(self.tablebase_dir, exist_ok=True)
        
        # Limit max_pieces to a reasonable value (3-5)
        max_pieces = min(max(3, max_pieces), 5)
        
        # URLs for different tablebases
        urls = {
            3: "https://tablebase.lichess.ovh/tables/3-piece/3-piece.zip",
            4: "https://tablebase.lichess.ovh/tables/4-piece/4-piece.zip",
            5: "https://tablebase.lichess.ovh/tables/5-piece/5-piece.zip",
            # 6-piece and 7-piece tablebases are very large (hundreds of GB)
        }
        
        for pieces in range(3, max_pieces + 1):
            try:
                url = urls.get(pieces)
                if not url:
                    logger.warning(f"No URL available for {pieces}-piece tablebases")
                    continue
                
                zip_path = os.path.join(self.tablebase_dir, f"{pieces}-piece.zip")
                
                logger.info(f"Downloading {pieces}-piece tablebases from {url}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    logger.info(f"Downloaded {pieces}-piece tablebases to {zip_path}")
                    
                    # Extract files
                    logger.info(f"Extracting {pieces}-piece tablebases...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.tablebase_dir)
                    
                    # Remove the zip file after extraction
                    os.remove(zip_path)
                    logger.info(f"Extracted and deleted {zip_path}")
                else:
                    logger.error(f"Failed to download {pieces}-piece tablebases: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading {pieces}-piece tablebases: {e}")
    
    def get_best_move(self, board):
        """
        Get the best move from the tablebase.
        
        Args:
            board: A chess.Board object.
            
        Returns:
            A chess.Move object, or None if no move is found.
        """
        if not self.enabled or not self.tablebase:
            return None
        
        try:
            # Check if position is in tablebase (piece count <= available pieces)
            piece_count = chess.popcount(board.occupied)
            
            if piece_count > self.detect_available_pieces():
                return None
            
            if not board.is_valid():
                logger.warning("Invalid board position")
                return None
            
            # Check if position is a tablebase position
            if not self.tablebase.probe_wdl(board):
                return None
            
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            
            # Evaluate each move using the tablebase
            best_move = None
            best_wdl = -2  # Worst possible WDL value
            
            for move in legal_moves:
                # Make the move
                board_copy = board.copy()
                board_copy.push(move)
                
                # If we're in the tablebase, probe it
                try:
                    # Probe DTZ (Distance To Zero - moves to conversion or win)
                    # Negate because score is from opponent's perspective after our move
                    wdl = -self.tablebase.probe_wdl(board_copy)
                    
                    # Update best move if this one is better
                    if wdl > best_wdl:
                        best_wdl = wdl
                        best_move = move
                except chess.syzygy.MissingTableError:
                    # Position not in tablebase
                    continue
            
            return best_move
        except Exception as e:
            logger.error(f"Error retrieving move from tablebase: {e}")
            return None
    
    def get_wdl(self, board):
        """
        Get Win-Draw-Loss value for the current position.
        
        Args:
            board: A chess.Board object.
            
        Returns:
            WDL value: 2 (win), 1 (winning), 0 (draw), -1 (losing), -2 (loss), None (not in tablebase)
        """
        if not self.enabled or not self.tablebase:
            return None
        
        try:
            # Check if position is in tablebase (piece count <= available pieces)
            piece_count = chess.popcount(board.occupied)
            
            if piece_count > self.detect_available_pieces():
                return None
            
            # Probe win-draw-loss value
            return self.tablebase.probe_wdl(board)
        except chess.syzygy.MissingTableError:
            return None
        except Exception as e:
            logger.error(f"Error probing tablebase: {e}")
            return None
    
    def close(self):
        """Close the tablebase."""
        if self.tablebase:
            self.tablebase.close()


class ChessExtras:
    """
    Combines opening book and endgame tablebase support for a chess bot.
    """
    def __init__(self, use_opening_book=True, use_tablebase=True, opening_book_path=None, tablebase_dir=None):
        """
        Initialize the chess extras.
        
        Args:
            use_opening_book: Whether to use the opening book.
            use_tablebase: Whether to use the endgame tablebase.
            opening_book_path: Path to the opening book file.
            tablebase_dir: Directory containing the tablebase files.
        """
        self.opening_book = None
        self.tablebase = None
        
        if use_opening_book:
            self.opening_book = OpeningBook(book_path=opening_book_path)
        
        if use_tablebase:
            self.tablebase = EndgameTablebase(tablebase_dir=tablebase_dir)
    
    def get_move(self, board_fen, primary_move_selector=None):
        """
        Get a move using opening book, tablebase, or the primary move selector.
        
        Args:
            board_fen: FEN string representing the board position.
            primary_move_selector: Function that takes a board and returns a move when no book/tablebase move is found.
            
        Returns:
            A chess.Move object, or None if no move is found.
        """
        # Convert FEN to chess.Board
        board = chess.Board(board_fen)
        
        # Try opening book first (early game)
        if self.opening_book:
            book_move = self.opening_book.get_move(board)
            if book_move:
                logger.info(f"Using opening book move: {book_move}")
                return book_move
        
        # Try tablebase next (endgame)
        if self.tablebase:
            tablebase_move = self.tablebase.get_best_move(board)
            if tablebase_move:
                logger.info(f"Using tablebase move: {tablebase_move}")
                return tablebase_move
        
        # Fall back to primary move selector
        if primary_move_selector:
            return primary_move_selector(board)
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.opening_book:
            self.opening_book.close()
        
        if self.tablebase:
            self.tablebase.close()


# Helper functions

def fen_to_board(fen):
    """Convert FEN string to chess.Board object."""
    return chess.Board(fen)


def board_to_fen(board):
    """Convert chess.Board object to FEN string."""
    return board.fen()


def move_to_uci(move):
    """Convert chess.Move object to UCI string."""
    return move.uci()


def uci_to_move(uci, board):
    """
    Convert UCI string to chess.Move object.
    
    Args:
        uci: UCI move string (e.g., "e2e4")
        board: chess.Board object to validate the move against
        
    Returns:
        chess.Move object or None if invalid
    """
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
        return None
    except ValueError:
        return None


def is_endgame(board):
    """
    Determine if the position is an endgame.
    
    Args:
        board: chess.Board object
        
    Returns:
        True if endgame, False otherwise
    """
    # Count major pieces (queen, rook)
    queens = chess.popcount(board.queens)
    rooks = chess.popcount(board.rooks)
    
    # If each side has at most one major piece, it's likely an endgame
    return queens + rooks <= 2


def is_opening(board):
    """
    Determine if the position is likely still in the opening phase.
    
    Args:
        board: chess.Board object
        
    Returns:
        True if opening, False otherwise
    """
    # Check number of moves
    if board.fullmove_number < 10:
        return True
    
    # Count developed pieces
    developed_pieces = 0
    
    # Knights and bishops out of their original squares indicate development
    if not board.knights & chess.BB_RANK_1:
        developed_pieces += chess.popcount(board.knights & chess.BB_WHITE)
    if not board.knights & chess.BB_RANK_8:
        developed_pieces += chess.popcount(board.knights & chess.BB_BLACK)
    if not board.bishops & chess.BB_RANK_1:
        developed_pieces += chess.popcount(board.bishops & chess.BB_WHITE)
    if not board.bishops & chess.BB_RANK_8:
        developed_pieces += chess.popcount(board.bishops & chess.BB_BLACK)
    
    # If less than 2 pieces are developed, likely still in opening
    return developed_pieces < 4


def get_move_category(fen):
    """
    Determine what category of move selection to use (opening, middlegame, endgame).
    
    Args:
        fen: FEN string representing the board position
        
    Returns:
        String: "opening", "middlegame", or "endgame"
    """
    board = chess.Board(fen)
    
    if is_opening(board):
        return "opening"
    elif is_endgame(board):
        return "endgame"
    else:
        return "middlegame"


# Example usage
if __name__ == "__main__":
    # Initialize the chess extras
    chess_extras = ChessExtras(use_opening_book=True, use_tablebase=True)
    
    # Test with starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = fen_to_board(starting_fen)
    
    # Check if opening book has a move
    if chess_extras.opening_book:
        book_move = chess_extras.opening_book.get_move(board)
        print(f"Opening book move for starting position: {book_move}")
    
    # Test with an endgame position (King+Rook vs King)
    endgame_fen = "8/8/8/8/8/K7/R7/7k w - - 0 1"
    endgame_board = fen_to_board(endgame_fen)
    
    # Check if tablebase has a move
    if chess_extras.tablebase:
        tablebase_move = chess_extras.tablebase.get_best_move(endgame_board)
        print(f"Tablebase move for endgame position: {tablebase_move}")
    
    # Clean up
    chess_extras.close() 