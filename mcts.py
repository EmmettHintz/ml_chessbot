import math
import numpy as np
import torch
import chess # Use python-chess

# Assuming model and utils are in the same directory or accessible
from model import AlphaZeroChessNet, device
# Import the updated utils functions
from utils import board_to_features, get_legal_moves_mask, move_to_index, index_to_move, POLICY_OUTPUT_SIZE

class Node:
    """Represents a node in the Monte Carlo Tree Search."""
    def __init__(self, parent=None, state_fen=None, prior_p=0.0, player_turn=chess.WHITE):
        """
        Initializes a new node.

        Args:
            parent: The parent node (Node or None for root).
            state_fen: The FEN string representing the board state.
            prior_p: Prior probability of reaching this node (from network policy).
            player_turn: Player whose turn it is (chess.WHITE or chess.BLACK).
        """
        self.parent = parent
        self.state_fen = state_fen
        self.player_turn = player_turn # Use chess.WHITE / chess.BLACK
        self.children = {}  # Maps move index to Node

        self.visit_count = 0
        self.total_action_value = 0.0 # Sum of values from simulations through this node (W)
        self.prior_p = prior_p # Prior probability P(s,a) from the network policy

        self._is_expanded = False
        self._board_instance = None # Cache board instance for efficiency

    def get_board(self):
        """Lazy loads and caches the python-chess Board object."""
        if self._board_instance is None and self.state_fen:
            try:
                 self._board_instance = chess.Board(self.state_fen)
            except ValueError:
                 print(f"Error: Invalid FEN string provided to Node: {self.state_fen}")
                 return None # Handle invalid FEN
        return self._board_instance

    def expand(self, policy_probs, value, legal_moves_mask):
        """
        Expands the node by creating children for all legal moves found in the mask.

        Args:
            policy_probs: Numpy array of policy probabilities from the network for this state.
            value: The value estimate (-1 to 1) from the network for this state.
            legal_moves_mask: Boolean mask indicating legal moves according to utils.py mapping.
        """
        if self._is_expanded:
            return value # Already expanded

        self._is_expanded = True
        current_board = self.get_board()
        if current_board is None:
             print(f"Error: Cannot expand node with no valid board state from FEN {self.state_fen}.")
             return 0

        next_player_turn = not self.player_turn # Flip turn (True/False)

        # Iterate through the moves deemed legal by the mask derived from our utils mapping
        for move_idx, is_legal in enumerate(legal_moves_mask):
            if is_legal > 0: # Mask might be float
                try:
                    move_uci = index_to_move(move_idx)
                    if move_uci and move_uci != "0000": # Ensure index_to_move returned a valid move
                         move = chess.Move.from_uci(move_uci)

                         # Double check legality with python-chess for safety
                         if move in current_board.legal_moves:
                             next_board_state = current_board.copy()
                             next_board_state.push(move)
                             child_fen = next_board_state.fen()
                             prior = policy_probs[move_idx]
                             self.children[move_idx] = Node(
                                 parent=self,
                                 state_fen=child_fen,
                                 prior_p=prior,
                                 player_turn=next_player_turn
                             )
                         # else:
                         #     # This indicates a mismatch between utils.get_legal_moves_mask and board.legal_moves
                         #     print(f"Warning: Move {move_uci} (idx {move_idx}) from mask is illegal in board {self.state_fen}. Skipping child.")
                    # else:
                         # print(f"Warning: index_to_move returned null/invalid for legal index {move_idx}. Skipping child.")

                except ValueError: # Handle potential errors from chess.Move.from_uci
                     print(f"Error parsing move UCI '{move_uci}' (index {move_idx}) during expansion. Skipping.")
                except Exception as e:
                     print(f"Error expanding node for move index {move_idx} from FEN {self.state_fen}: {e}")

        # Return the value estimate from the network for backpropagation
        # The value is from the perspective of the current node's player
        return value

    def select_child(self, c_puct):
        """
        Selects the child node with the highest PUCT score.

        Args:
            c_puct: Exploration constant.

        Returns:
            The selected child node (Node) and the move index leading to it.
            Returns (None, -1) if no children exist or selection fails.
        """
        if not self.children:
            return None, -1 # Cannot select if no children

        best_score = -float('inf')
        best_move_idx = -1
        best_child = None

        # Calculate total visits of siblings once
        total_parent_visits = self.visit_count
        sqrt_total_parent_visits = math.sqrt(max(1, total_parent_visits))

        for move_idx, child in self.children.items():
            # Q(s, a): Mean action value (value from the child's perspective)
            q_value = child.total_action_value / (child.visit_count + 1e-8)

            # U(s, a): Exploration term
            u_value = c_puct * child.prior_p * (sqrt_total_parent_visits / (1 + child.visit_count))

            # PUCT score - negate q_value as it's from opponent's perspective
            score = -q_value + u_value

            if score > best_score:
                best_score = score
                best_move_idx = move_idx
                best_child = child

        return best_child, best_move_idx

    def backpropagate(self, value):
        """
        Backpropagates the simulation result up the tree.

        Args:
            value: The value estimate (-1 to 1) from the simulation/evaluation.
                   This value is from the perspective of the player whose turn it was
                   at the node where the evaluation occurred.
        """
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            # Add value, ensuring perspective matches the node's player turn
            node.total_action_value += current_value
            node = node.parent
            current_value *= -1 # Flip value for the parent's perspective

    def is_leaf(self):
        """Checks if the node is a leaf node (not yet expanded)."""
        return not self._is_expanded

    def is_terminal(self):
        """Checks if the node represents a terminal game state using python-chess."""
        board = self.get_board()
        return board is not None and board.is_game_over()

    def get_outcome(self):
        """
        Gets the game outcome from python-chess.
        Returns: 1 for white win, -1 for black win, 0 for draw, None if not over.
        """
        board = self.get_board()
        if board is None or not board.is_game_over():
            return None
        outcome = board.outcome()
        if outcome is None: # Should not happen if is_game_over is true
             return 0
        if outcome.winner == chess.WHITE:
            return 1
        elif outcome.winner == chess.BLACK:
            return -1
        else:
            return 0 # Draw

    @property
    def mean_action_value(self):
        """Calculates the mean action value (Q) for this node."""
        if self.visit_count == 0:
            return 0.0
        # W(s) / N(s)
        return self.total_action_value / self.visit_count


class MCTS:
    """Manages the Monte Carlo Tree Search process using python-chess."""
    def __init__(self, model: AlphaZeroChessNet, c_puct=1.0, num_simulations=100):
        """
        Initializes MCTS.

        Args:
            model: The neural network model (AlphaZeroChessNet).
            c_puct: Exploration constant.
            num_simulations: Number of simulations to run per move.
        """
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run_simulations(self, root_fen):
        """
        Runs the MCTS simulations starting from the root FEN.

        Args:
            root_fen: The FEN string of the current board state.

        Returns:
            The root node after simulations.
        """
        try:
            root_board = chess.Board(root_fen)
            player_turn = root_board.turn # True for White, False for Black
        except ValueError:
             print(f"Error: Invalid root FEN provided to run_simulations: {root_fen}")
             return None # Cannot proceed with invalid FEN

        root = Node(state_fen=root_fen, player_turn=player_turn)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. Select
            while node._is_expanded and not node.is_terminal():
                selected_child, _ = node.select_child(self.c_puct)
                if selected_child is None:
                    # print(f"Warning: select_child returned None for node {node.state_fen}. Breaking selection.")
                    break
                node = selected_child
                search_path.append(node)

            if node is None: # Should only happen if root has no children
                 print(f"Error: MCTS selection failed, node is None. Skipping simulation for {root_fen}.")
                 continue

            # 2. Evaluate leaf node or terminal node
            value = 0.0
            if node.is_terminal():
                # Game ended, get the result relative to the *current node's player*
                outcome = node.get_outcome()
                # Value should be from the perspective of the player TO MOVE at the terminal node
                value = outcome if node.player_turn == chess.WHITE else -outcome
            else:
                # 3. Expand & Evaluate using Network
                # Get policy and value from the network for the leaf node
                # The network prediction is always from White's perspective? Assume yes.
                # We need policy probs (numpy array) and value (scalar) for node.state_fen
                raw_policy_logits, raw_value_output = self.model.predict(node.state_fen) # Use raw model output

                # Convert policy tensor to numpy
                raw_policy_logits_np = raw_policy_logits.cpu().numpy() # Ensure it's on CPU before numpy conversion

                # Post-process policy: apply legality mask and softmax
                legal_mask = get_legal_moves_mask(node.state_fen)
                masked_policy = raw_policy_logits_np * legal_mask # Now numpy * numpy
                # Avoid division by zero if no legal moves in mask (should not happen in non-terminal state)
                sum_masked_policy = np.sum(masked_policy) # Should work now
                if sum_masked_policy > 1e-8:
                    policy_probs = masked_policy / sum_masked_policy
                else:
                    print(f"Warning: No legal moves found in policy mask for FEN {node.state_fen}. Assigning uniform probability.")
                    # Assign uniform probability to truly legal moves if mask failed
                    board = node.get_board()
                    policy_probs = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
                    if board and board.legal_moves:
                         num_legal = len(list(board.legal_moves))
                         prob = 1.0 / num_legal if num_legal > 0 else 0
                         for move in board.legal_moves:
                            move_idx = move_to_index(move.uci())
                            if move_idx != -1:
                                 policy_probs[move_idx] = prob
                    # If still zero, something is wrong
                    if np.sum(policy_probs) < 1e-8:
                         policy_probs = legal_mask # Fallback to raw mask if uniform assignment failed


                # Network value is typically from white's perspective
                network_value = raw_value_output.item()

                # Expand the node using the processed policy_probs
                # The expand function returns the network_value for backpropagation
                value = node.expand(policy_probs, network_value, legal_mask)

                # The value for backpropagation needs to be from the perspective of the player
                # whose turn it was at the expanded node.
                # If network value is from white's perspective:
                value = network_value if node.player_turn == chess.WHITE else -network_value

            # 4. Backpropagate
            # The value should be from the perspective of the player whose turn it was at the leaf/terminal node.
            node.backpropagate(value)

        return root

    def get_action_probs(self, root_fen, temperature=1.0):
        """
        Runs MCTS and returns the improved policy distribution.

        Args:
            root_fen: The FEN string of the current board state.
            temperature: Controls exploration in the returned policy (0=greedy, >0=stochastic).

        Returns:
            A numpy array representing the probability distribution over moves (size POLICY_OUTPUT_SIZE),
            and the estimated value of the root state.
        """
        root = self.run_simulations(root_fen)
        if root is None:
             # Simulation failed (e.g., invalid FEN)
             return np.zeros(POLICY_OUTPUT_SIZE), 0.0

        policy_probs = np.zeros(POLICY_OUTPUT_SIZE)

        if not root.children: # No legal moves (checkmate/stalemate)
            return policy_probs, root.mean_action_value

        # Use child visit counts to determine policy
        child_visits = np.array([child.visit_count for child in root.children.values()])
        move_indices = list(root.children.keys())

        if temperature == 0: # Greedy selection
            best_move_idx_in_list = np.argmax(child_visits)
            best_move_global_idx = move_indices[best_move_idx_in_list]
            policy_probs[best_move_global_idx] = 1.0
        else:
            # Apply temperature
            visits_temp = child_visits**(1.0 / temperature)
            visits_sum = np.sum(visits_temp)
            if visits_sum > 1e-8:
                 normalized_probs = visits_temp / visits_sum
                 for idx, prob in zip(move_indices, normalized_probs):
                     if 0 <= idx < POLICY_OUTPUT_SIZE:
                         policy_probs[idx] = prob
            # else: Handle case where all visit counts are zero (shouldn't happen with >0 sims)

        # Return the probability distribution and the root node's value estimate
        # Value is W(s_root) / N(s_root)
        root_value = root.mean_action_value
        return policy_probs, root_value

# Note: Example Usage removed as it requires a trained model and specific setup 