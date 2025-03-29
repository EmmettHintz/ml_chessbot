import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from kaggle_environments import make # Assuming not needed if we don't generate games this way
import chess # Replace Chessnut with python-chess
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Ensure we're using the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS for board representation (adjust if needed based on AlphaZero paper/impl)
# 18 channels: 6 piece types x 2 colors + 6 history planes (optional)
PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
COLOR_OFFSET = 6  # Black pieces start at index 6
TOTAL_PIECE_PLANES = 12

# Other planes (adjust based on model architecture)
# Example: 1 plane for whose turn, 1 for total move count, 1 for P1 castling K, 1 for P1 Q, 1 for P2 K, 1 for P2 Q, 1 for en passant
TURN_PLANE = 12
# Define indices for the 4 castling planes directly
WHITE_KS_CASTLE_PLANE = 13
WHITE_QS_CASTLE_PLANE = 14
BLACK_KS_CASTLE_PLANE = 15
BLACK_QS_CASTLE_PLANE = 16
EN_PASSANT_PLANE = 17
NUM_INPUT_CHANNELS = 18 # Make sure this matches model definition

# Move representation constants (AlphaZero uses 4672 for chess)
# Simplified 8x8x73 representation is often used:
# 64 squares * 56 queen-like moves + 64 squares * 8 knight moves + 64 squares * 9 underpromotions = 4672
# Alternatively, use a flat list of all possible UCI moves (1968 as per your model?)
POLICY_OUTPUT_SIZE = 1968 # Ensure this matches model definition

# Map pieces to their corresponding planes
def piece_to_plane(piece):
    plane_index = PIECE_PLANES[piece.piece_type]
    if piece.color == chess.BLACK:
        plane_index += COLOR_OFFSET
    return plane_index

def board_to_features(fen):
    """
    Convert FEN board representation to feature planes (numpy array).
    Returns a numpy array with shape (NUM_INPUT_CHANNELS, 8, 8).
    Adjust based on the specific input representation required by the model.
    """
    board = chess.Board(fen)
    features = np.zeros((NUM_INPUT_CHANNELS, 8, 8), dtype=np.float32)

    # 1. Piece positions (Planes 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_to_plane(piece)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            features[plane, 7 - rank, file] = 1 # Rank 0 is bottom row in numpy

    # 2. Player Turn (Plane 12)
    if board.turn == chess.WHITE:
        features[TURN_PLANE, :, :] = 1
    else: # Black's turn
        features[TURN_PLANE, :, :] = 0 # Or 0, depends on convention

    # 3. Castling Rights (Planes 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        features[WHITE_KS_CASTLE_PLANE, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        features[WHITE_QS_CASTLE_PLANE, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        features[BLACK_KS_CASTLE_PLANE, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        features[BLACK_QS_CASTLE_PLANE, :, :] = 1

    # 4. En Passant (Plane 17)
    if board.ep_square:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        features[EN_PASSANT_PLANE, 7 - rank, file] = 1

    # Add other planes if needed (e.g., move counts, repetition history)

    return features


# --- Move Indexing ---
# This needs to perfectly match the policy head output of your model.
# The common AlphaZero approach maps every possible *legal* move type
# originating from each square to a unique index.
# For chess, this often results in 4672 dimensions.
# Your current model uses 1968. We need to define a consistent mapping.
# Let's assume a simple flat UCI list approach for 1968, though this might not be optimal.
# We need a definitive list of all 1968 possible UCI strings the model predicts.
# Lichess provides a list often used: https://github.com/lichess-org/chess-engine-eval/blob/master/uci_corpus/uci-moves-960.txt
# (This is for Chess960, standard chess is a subset)
# For now, we'll define a placeholder mapping. THIS IS LIKELY THE CRITICAL PART TO GET RIGHT.

# Placeholder: Generate a fixed list of potential UCI moves (needs refinement)
# This should ideally cover all standard moves + promotions
_POSSIBLE_MOVES_UCI = []
def _generate_possible_moves():
    global _POSSIBLE_MOVES_UCI
    if _POSSIBLE_MOVES_UCI: return

    # All possible moves from standard starting board + several iterations of common positions
    # Use explicit board positions to generate more likely moves in actual games
    all_moves = set()
    
    # Define important chess moves that should always be included
    important_moves = [
        # Common opening moves
        "e2e4", "d2d4", "c2c4", "g1f3", "e7e5", "d7d5", "c7c5", "g8f6",
        "b1c3", "b8c6", "f1c4", "f8c5", "e1g1", "e8g8",  # castling is O-O in UCI
        # Common pawn moves
        "e2e3", "d2d3", "c2c3", "b2b3", "a2a3", "g2g3", "h2h3", "f2f3",
        "e7e6", "d7d6", "c7c6", "b7b6", "a7a6", "g7g6", "h7h6", "f7f6", 
        # Common knight moves
        "g1f3", "b1c3", "g8f6", "b8c6", "f3d4", "c3d5", "f6d7", "c6d4",
        # Common bishop moves
        "c1e3", "f1c4", "c8e6", "f8c5",
        # Common rook moves
        "a1c1", "h1f1", "a8c8", "h8f8",
        # Common queen moves
        "d1d2", "d1c2", "d1e2", "d8d7", "d8c7", "d8e7"
    ]
    
    # Generate moves from several common positions
    starting_positions = [
        chess.Board(), # Standard starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), # After e4
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"), # After e4 e5
        chess.Board("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"), # After d4 d5
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2"), # After c4 e5
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"), # After e4 e5 Nf3
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"), # After e4 e5 Nf3 Nc6
    ]
    
    # Ensure important moves are in our set
    for move in important_moves:
        all_moves.add(move)
    
    # Generate legal moves from each position and add them to our set
    for board in starting_positions:
        for move in board.legal_moves:
            all_moves.add(move.uci())
            # Make the move and add legal responses
            board_copy = board.copy()
            board_copy.push(move)
            for response in board_copy.legal_moves:
                all_moves.add(response.uci())
    
    # Now add all possible moves between any squares
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq: continue
            all_moves.add(chess.Move(from_sq, to_sq).uci())
            
            # Add special moves - promotions
            # For white pawns
            if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    all_moves.add(chess.Move(from_sq, to_sq, promotion=prom).uci())
            # For black pawns
            if chess.square_rank(from_sq) == 1 and chess.square_rank(to_sq) == 0:
                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    all_moves.add(chess.Move(from_sq, to_sq, promotion=prom).uci())
    
    # Print info before prioritization
    original_size = len(all_moves)
    print(f"Generated {original_size} potential moves")
    
    # Prioritize moves - first ensure important moves are included
    prioritized_moves = list(important_moves)
    
    # Then add remaining moves that aren't in the important list
    remaining_moves = list(all_moves - set(important_moves))
    
    # Sort remaining moves to ensure deterministic behavior
    remaining_moves.sort()
    
    # Combine lists - important moves first, then the rest
    combined_moves = prioritized_moves + remaining_moves
    
    # Pad or truncate to POLICY_OUTPUT_SIZE
    if len(combined_moves) < POLICY_OUTPUT_SIZE:
        combined_moves.extend(["0000"] * (POLICY_OUTPUT_SIZE - len(combined_moves)))
    elif len(combined_moves) > POLICY_OUTPUT_SIZE:
        print(f"Warning: Generated {len(combined_moves)} potential moves, but policy size is {POLICY_OUTPUT_SIZE}. Truncating.")
        # Only truncate from the remaining moves, not from important moves
        if len(prioritized_moves) > POLICY_OUTPUT_SIZE:
            # Too many important moves, have to truncate some
            combined_moves = combined_moves[:POLICY_OUTPUT_SIZE]
        else:
            # Keep all important moves, truncate only from remaining
            combined_moves = prioritized_moves + remaining_moves[:(POLICY_OUTPUT_SIZE - len(prioritized_moves))]
    
    _POSSIBLE_MOVES_UCI = combined_moves
    
    # Print statistics
    print(f"Final move list has {len(_POSSIBLE_MOVES_UCI)} moves")
    
    # Debug - checking that common openings are covered
    common_moves = ["e2e4", "d2d4", "e7e5", "d7d5", "g1f3", "b8c6", "f1c4"]
    for move in common_moves:
        if move in _POSSIBLE_MOVES_UCI:
            print(f"Common move {move} is at index {_POSSIBLE_MOVES_UCI.index(move)}")
        else:
            print(f"WARNING: Common move {move} is NOT in our move list!")


_generate_possible_moves()
_MOVE_TO_INDEX_MAP = {move: i for i, move in enumerate(_POSSIBLE_MOVES_UCI)}
_INDEX_TO_MOVE_MAP = {i: move for i, move in enumerate(_POSSIBLE_MOVES_UCI)}

def move_to_index(move_uci):
    """
    Convert a chess move in UCI format (e.g., 'e2e4') to its index
    in the predefined list _POSSIBLE_MOVES_UCI.
    """
    # Ensure the list is generated
    if not _POSSIBLE_MOVES_UCI:
        _generate_possible_moves()

    return _MOVE_TO_INDEX_MAP.get(move_uci, -1) # Return -1 for unknown moves

def index_to_move(index):
    """
    Convert a move index back to UCI format using the predefined list.
    """
     # Ensure the list is generated
    if not _POSSIBLE_MOVES_UCI:
        _generate_possible_moves()

    if 0 <= index < len(_POSSIBLE_MOVES_UCI):
        return _INDEX_TO_MOVE_MAP[index]
    else:
        print(f"Warning: index_to_move received invalid index {index}")
        return "0000" # Return null move for invalid index


def get_legal_moves_mask(board_fen):
    """
    Create a binary mask of legal moves for the given board FEN.
    The mask corresponds to the indices in _POSSIBLE_MOVES_UCI.
    1 indicates a legal move, 0 indicates illegal.
    """
    board = chess.Board(board_fen)
    mask = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32) # Use float for potential probabilities
    
    # Count for debugging
    found_moves = 0
    total_legal_moves = 0
    
    for move in board.legal_moves:
        total_legal_moves += 1
        move_uci = move.uci()
        move_idx = move_to_index(move_uci)
        if move_idx != -1: # If the legal move is in our known list
            mask[move_idx] = 1.0
            found_moves += 1
        # else:
            # print(f"Warning: Legal move {move_uci} not found in predefined move list.")
    
    # Debug - count how many legal moves were found in our list
    coverage_pct = (found_moves / total_legal_moves * 100) if total_legal_moves > 0 else 0
    
    # Only warn if we have a real issue (less than 80% coverage)
    if found_moves < total_legal_moves and coverage_pct < 80:
        print(f"Warning: Found {found_moves}/{total_legal_moves} legal moves ({coverage_pct:.1f}%) for FEN {board_fen}")
    
    # If the mask has no legal moves found in our list, return all zeros.
    if np.sum(mask) == 0 and total_legal_moves > 0:
        print(f"Warning: No legal moves from FEN {board_fen} found in the predefined move list.")
    
    return mask

# --- Evaluation and Plotting (Keep or adapt as needed) ---

def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Policy Loss
    axs[0, 0].plot(history['policy_loss'], label='Train Policy Loss')
    axs[0, 0].plot(history['val_policy_loss'], label='Val Policy Loss')
    axs[0, 0].set_title('Policy Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Value Loss
    axs[0, 1].plot(history['value_loss'], label='Train Value Loss')
    axs[0, 1].plot(history['val_value_loss'], label='Val Value Loss')
    axs[0, 1].set_title('Value Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('MSE')
    axs[0, 1].legend()

    # Policy Accuracy
    axs[1, 0].plot(history['policy_accuracy'], label='Train Policy Accuracy')
    axs[1, 0].plot(history['val_policy_accuracy'], label='Val Policy Accuracy')
    axs[1, 0].set_title('Policy Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()

    # Value Error (if calculated similarly)
    # You might need to adjust this based on how value accuracy/error is logged
    # axs[1, 1].plot(history.get('value_accuracy', []), label='Train Value Accuracy')
    # axs[1, 1].plot(history.get('val_value_accuracy', []), label='Val Value Accuracy')
    axs[1, 1].set_title('Value Prediction (Placeholder)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Metric')
    axs[1, 1].legend()


    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

# --- Memory Analysis (Keep or adapt) ---
# This part seems specific to the DualStreamMemoryNetwork, might not apply directly to AlphaZeroChessNet

# def analyze_memory_patterns(model, num_components=2, num_clusters=5):
#     """Analyze memory patterns using PCA and K-Means"""
#     model.eval()
#     # Assuming memory is stored in model.memory or accessible via a method
#     # This needs adjustment based on the actual model structure
#     memory_states = []
#     if hasattr(model, 'memory') and model.memory is not None:
#          memory_states = model.memory.cpu().detach().numpy()
#     elif hasattr(model, 'get_memory_states'):
#          memory_states = model.get_memory_states() # Needs implementation in model
#     else:
#         print("Warning: Cannot access memory states for analysis.")
#         return None, None

#     if not memory_states or len(memory_states) < num_components:
#          print("Warning: Not enough memory states for analysis.")
#          return None, None

#     # Reshape if necessary (e.g., if memory is [batch, seq, features])
#     if len(memory_states.shape) > 2:
#          memory_states = memory_states.reshape(-1, memory_states.shape[-1])

#     # Apply PCA
#     pca = PCA(n_components=num_components)
#     memory_pca = pca.fit_transform(memory_states)

#     # Apply K-Means
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10) # Set n_init explicitly
#     clusters = kmeans.fit_predict(memory_pca)

#     # Plot results
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(memory_pca[:, 0], memory_pca[:, 1], c=clusters, cmap='viridis')
#     plt.title('Memory State Clusters after PCA')
#     plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
#     plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
#     plt.legend(handles=scatter.legend_elements()[0], title="Clusters")
#     plt.savefig("memory_clusters.png")
#     plt.show()

#     return memory_pca, clusters


# def visualize_attention(model, board_fen):
#     """Visualize attention weights for a given board state"""
#     model.eval()
#     features = board_to_features(board_fen)
#     features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and seq dims? [1, 1, C, H, W]? Adjust shape

#     # Need to adapt this based on how the model returns attention weights
#     # Assuming model returns (policy, value, memory, attention_weights)
#     # Or has a method to get attention weights
#     attention_weights = None
#     if hasattr(model, 'get_attention_weights'):
#         _, _, _, attention_weights = model(features_tensor) # Adjust call signature
#     else:
#         # Attempt to access attention if it's a stored attribute (less likely)
#         # Or modify the model's forward pass to return it
#         print("Warning: Cannot retrieve attention weights from the model.")
#         return None


#     if attention_weights is None:
#         return None

#     # Process attention weights (example: average over heads/layers if needed)
#     # Shape might be [batch, seq, heads, query_len, key_len] -> needs processing
#     # Assuming weights are related to board squares [8x8] or [64x64] after processing
#     processed_weights = attention_weights.squeeze().cpu().detach().numpy() # Example processing
#     if processed_weights.ndim > 2:
#          # Simple averaging if multiple heads/layers present
#          processed_weights = np.mean(processed_weights, axis=tuple(range(processed_weights.ndim - 2)))


#     if processed_weights.shape == (64, 64): # Attention between squares
#          # Display as heatmap
#          plt.figure(figsize=(8, 8))
#          plt.imshow(processed_weights, cmap='hot', interpolation='nearest')
#          plt.title(f'Attention Map for FEN: {board_fen}')
#          plt.xlabel("Key Squares (Index)")
#          plt.ylabel("Query Squares (Index)")
#          plt.colorbar()
#          plt.savefig("attention_map.png")
#          plt.show()
#          return processed_weights
#     elif processed_weights.shape == (8, 8): # Attention weights per square
#          plt.figure(figsize=(8, 8))
#          plt.imshow(processed_weights, cmap='hot', interpolation='nearest')
#          plt.title(f'Attention Weights per Square for FEN: {board_fen}')
#          plt.xticks(np.arange(8), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
#          plt.yticks(np.arange(8), reversed(range(1, 9))) # Board coordinates
#          plt.colorbar()
#          plt.savefig("attention_weights.png")
#          plt.show()
#          return processed_weights
#     else:
#         print(f"Warning: Unexpected attention weights shape: {processed_weights.shape}")
#         return None


# --- Old functions using Chessnut (to be removed or replaced) ---
# def evaluate_model(...): (needs complete rewrite with python-chess)
# def generate_sample_games(...): (needs rewrite with python-chess)
# def prepare_batch(...): (needs rewrite with python-chess moves/values)
