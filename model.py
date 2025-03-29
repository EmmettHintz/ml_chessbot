import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import chess # Use python-chess instead of Chessnut
from utils import board_to_features, move_to_index, index_to_move, get_legal_moves_mask

# Ensure we're using the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- New Residual Block ---
class ResidualBlock(nn.Module):
    """
    A residual block as used in AlphaZero.
    Input -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, num_filters=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Skip connection
        out = F.relu(out)
        return out

# --- New AlphaZero-style Network ---
class AlphaZeroChessNet(nn.Module):
    """
    AlphaZero-style network using Residual Blocks.
    Consists of an initial convolutional block, a stack of residual blocks,
    and separate policy and value heads.
    """
    def __init__(self, input_channels=18, num_filters=256, num_res_blocks=19, policy_output_size=1968):
        super(AlphaZeroChessNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_output_size = policy_output_size

        # Initial convolutional block
        self.initial_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_output_size) # 8x8 board size

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256) # 8x8 board size
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        Input x shape: (batch_size, channels, height, width) e.g., (N, 18, 8, 8)
        """
        # Process input features if needed (assuming utils.board_to_features provides correct shape)
        if len(x.shape) == 5:  # [batch, seq, 8, 8, channels] -> Used by old network
             # Reshape for conv: [batch * seq, channels, 8, 8]
             batch_size, seq_len = x.shape[0], x.shape[1]
             x = x.permute(0, 1, 4, 2, 3) # [batch, seq, channels, 8, 8]
             x = x.reshape(batch_size * seq_len, self.input_channels, 8, 8)
        elif len(x.shape) == 4 and x.shape[1] != self.input_channels : # [batch, 8, 8, channels] -> User might provide this
             x = x.permute(0, 3, 1, 2) # [batch, channels, 8, 8]
        # Else: Assume x is already [batch, channels, 8, 8]

        # Initial block
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1) # Flatten
        policy_logits = self.policy_fc(policy)
        # Note: Softmax is applied outside the model (usually in the loss function or post-processing)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1) # Flatten
        value = F.relu(self.value_fc1(value))
        value_output = torch.tanh(self.value_fc2(value)) # Output between -1 and 1

        # If input was sequential, reshape output? - For now, assume batch processing
        # This needs clarification based on how MCTS will use batched states if needed.
        # If batch_size * seq_len was used:
        # policy_logits = policy_logits.view(batch_size, seq_len, -1)
        # value_output = value_output.view(batch_size, seq_len, 1)

        return policy_logits, value_output

    def predict(self, board_fen):
        """
        Get raw policy logits and value prediction for a FEN string.
        Used by MCTS.
        """
        self.eval() # Set model to evaluation mode
        features = board_to_features(board_fen)
        # Ensure features are in the correct format (C, H, W)
        if len(features.shape) == 3 and features.shape[0] != self.input_channels:
             features = np.transpose(features, (2, 0, 1))

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension

        # Get model prediction
        with torch.no_grad():
             # Ensure input tensor has shape [1, channels, 8, 8]
             if features_tensor.shape != (1, self.input_channels, 8, 8):
                  raise ValueError(f"Input tensor shape mismatch. Expected (1, {self.input_channels}, 8, 8), got {features_tensor.shape}. Check board_to_features output.")
             
             policy_logits, value_output = self(features_tensor)

        # Return raw logits and value, remove batch dimension
        return policy_logits.squeeze(0), value_output.squeeze(0)

    def predict_move(self, board_fen, apply_legality_mask=True):
        """
        Predict the best move for a given board position using the AlphaZero network.
        Mainly used for inference/evaluation. Includes legality masking and softmax.

        Args:
            board_fen: FEN string representing the board
            apply_legality_mask: Whether to filter illegal moves

        Returns:
            best_move: The predicted best move in UCI format
            move_probabilities: Probabilities for all moves (after softmax and optional masking)
            position_value: Evaluation of the position [-1, 1]
        """
        self.eval() # Set model to evaluation mode

        # Use the predict method to get raw outputs
        policy_logits, value_output = self.predict(board_fen) 
        value = value_output.item() # Get scalar value

        # Apply mask for legal moves if requested
        if apply_legality_mask:
            try:
                # Use python-chess to get legal moves mask
                legal_moves_mask = get_legal_moves_mask(board_fen)
                legal_moves_mask_tensor = torch.tensor(legal_moves_mask, dtype=torch.bool).to(device) # Use boolean mask

                # Apply mask by setting illegal move logits to a very small number (-inf)
                policy_logits[~legal_moves_mask_tensor] = -float('inf')

            except Exception as e:
                 print(f"Warning: Could not apply legality mask for FEN '{board_fen}'. Error: {e}")
                 # Decide how to handle this: maybe return error, or proceed without mask?
                 # Proceeding without mask for now.
                 pass

        # Get move probabilities using softmax
        move_probabilities = F.softmax(policy_logits, dim=-1)

        # Get best move
        # Ensure there's at least one legal move, otherwise argmax on -inf might be problematic
        if apply_legality_mask and torch.all(policy_logits == -float('inf')):
             print(f"Warning: No legal moves found or predicted for FEN '{board_fen}'. This indicates a checkmate, stalemate, or error.")
             # Handle terminal state - perhaps return None or a special indicator
             best_move_idx = -1 # Or handle appropriately
             best_move = None
        else:
             best_move_idx = torch.argmax(move_probabilities).item()
             try:
                 best_move = index_to_move(best_move_idx)
             except Exception as e:
                 print(f"Error converting index {best_move_idx} to move: {e}")
                 best_move = None # Handle potential errors in index_to_move

        return best_move, move_probabilities.cpu().numpy(), value

# --- Keep utility functions (or ensure they are correctly imported) ---
# Example: If board_to_features, move_to_index, index_to_move, get_legal_moves_mask
# were defined in this file previously, they should remain or be imported.
# Assuming they are correctly imported from 'utils' as per the top lines.

# Example instantiation (optional, for testing)
# if __name__ == '__main__':
#     model = AlphaZeroChessNet().to(device)
#     print(model)
#     # Example dummy input
#     dummy_input = torch.randn(4, 18, 8, 8).to(device) # Batch of 4
#     policy, value = model(dummy_input)
#     print("Policy shape:", policy.shape) # Should be [4, 1968]
#     print("Value shape:", value.shape)   # Should be [4, 1]

#     # Example prediction
#     initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#     best_move, probs, val = model.predict_move(initial_fen)
#     print(f"Initial FEN Best Move: {best_move}, Value: {val:.4f}")
