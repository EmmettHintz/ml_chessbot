import os
import numpy as np
import torch
import time
import argparse
from datetime import datetime
import json
import random
from tqdm import tqdm
import chess # Use python-chess

from model import AlphaZeroChessNet, device
from mcts import MCTS
from utils import index_to_move # Use updated utils

class Evaluator:
    """
    Handles evaluation of chess models by playing games between them using python-chess.
    """
    def __init__(
        self,
        model1,
        model2,
        num_games=100,
        num_simulations=100,
        c_puct=1.0,
        temperature=0.0, # Use temperature=0 for deterministic evaluation
        max_moves=512,
        results_dir="evaluation_results",
        swap_sides=True,
    ):
        """ Initialize the evaluator. Args are the same, implementation uses python-chess. """
        self.model1 = model1
        self.model2 = model2
        self.num_games = num_games
        # MCTS instances should already use python-chess
        self.mcts1 = MCTS(self.model1, c_puct=c_puct, num_simulations=num_simulations)
        self.mcts2 = MCTS(self.model2, c_puct=c_puct, num_simulations=num_simulations)
        self.temperature = temperature # Typically 0 for eval
        self.max_moves = max_moves
        self.results_dir = results_dir
        self.swap_sides = swap_sides

        os.makedirs(results_dir, exist_ok=True)

    def play_game(self, model1_plays_white=True, game_id=0, verbose=False):
        """
        Play a single game between the two models using python-chess.

        Args:
            model1_plays_white: Whether model1 plays as white.
            game_id: Game identifier for logging.
            verbose: Whether to print game progress.

        Returns:
            Result from model1's perspective (1=win, -1=loss, 0=draw), game_data dict.
        """
        board = chess.Board() # Starting position
        moves = []
        states = [] # Store FENs

        white_mcts = self.mcts1 if model1_plays_white else self.mcts2
        black_mcts = self.mcts2 if model1_plays_white else self.mcts1

        move_count = 0
        start_time = time.time()

        while not board.is_game_over(claim_draw=True) and move_count < self.max_moves:
            board_fen = board.fen()
            states.append(board_fen)

            current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts

            # Use MCTS to get move probabilities (temperature=0 for greedy selection)
            action_probs, _ = current_mcts.get_action_probs(board_fen, self.temperature)

            # Select best move (deterministic due to temperature=0)
            valid_move_indices = np.where(action_probs > 0)[0]
            if len(valid_move_indices) == 0:
                print(f"CRITICAL Error: MCTS returned no valid moves in evaluation game {game_id} for FEN: {board_fen}. Aborting.")
                # Determine result based on current state if possible (e.g., stalemate check)
                # For now, treat as an error/abort -> potentially draw?
                return 0, { "error": "MCTS failed", "game_id": game_id, "final_fen": board_fen } # Indicate error

            move_idx = valid_move_indices[np.argmax(action_probs[valid_move_indices])]
            move_uci = index_to_move(move_idx) # Use updated utils

            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                     print(f"CRITICAL Error: MCTS suggested illegal move {move_uci} in eval game {game_id}. Aborting.")
                     return 0, { "error": "Illegal move suggested", "game_id": game_id, "final_fen": board_fen }
            except ValueError:
                 print(f"CRITICAL Error: Invalid move UCI '{move_uci}' (index {move_idx}) in eval game {game_id}. Aborting.")
                 return 0, { "error": "Invalid UCI", "game_id": game_id, "final_fen": board_fen }

            moves.append(move.uci())

            if verbose and move_count % 10 == 0:
                player = "M1(W)" if model1_plays_white else "M2(W)"
                if board.turn == chess.BLACK:
                    player = "M1(B)" if not model1_plays_white else "M2(B)"
                print(f"Eval {game_id}, Mv {move_count}: {player} -> {move.uci()}")

            board.push(move)
            move_count += 1

        # Determine the result
        final_result = 0 # 1: White wins, -1: Black wins, 0: Draw
        forced_draw = False
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE:
                final_result = 1
            elif outcome.winner == chess.BLACK:
                final_result = -1
            else:
                final_result = 0
        elif move_count >= self.max_moves:
            final_result = 0 # Draw due to move limit
            forced_draw = True

        # Convert result to model1's perspective
        model1_result = 0
        if final_result == 1: # White won
            model1_result = 1 if model1_plays_white else -1
        elif final_result == -1: # Black won
            model1_result = -1 if model1_plays_white else 1
        # else: draw (model1_result remains 0)

        game_time = time.time() - start_time
        outcome_str = outcome.termination.name if outcome else ("MAX_MOVES_DRAW" if forced_draw else "UNKNOWN")

        if verbose:
            result_desc = "M1 Wins" if model1_result == 1 else ("M2 Wins" if model1_result == -1 else "Draw")
            print(f"Eval {game_id} finished in {move_count} moves ({game_time:.1f}s). Outcome: {outcome_str}. Result: {result_desc}")

        game_data = {
            "game_id": game_id,
            "model1_plays_white": model1_plays_white,
            "moves": moves,
            "result_model1": model1_result,
            "final_result_abs": final_result, # 1=White, -1=Black, 0=Draw
            "outcome": outcome_str,
            "move_count": move_count,
            "game_time": game_time,
            "final_fen": board.fen()
        }

        return model1_result, game_data

    def evaluate(self, verbose=True):
        """
        Evaluate model1 (challenger) against model2 (current best).
        (Logic mostly unchanged, uses updated play_game)
        """
        self.model1.eval()
        self.model2.eval()

        model1_wins = 0
        model2_wins = 0
        draws = 0
        errors = 0
        games_data = []

        total_games = self.num_games
        progress_bar = tqdm(total=total_games, desc="Evaluating Models") if verbose else None

        for game_id in range(total_games):
            model1_plays_white = game_id % 2 == 0 if self.swap_sides else True

            result, game_data = self.play_game(model1_plays_white, game_id, verbose=False) # Less verbose game play
            games_data.append(game_data)

            if "error" in game_data:
                 errors += 1
            elif result == 1:
                model1_wins += 1
            elif result == -1:
                model2_wins += 1
            else:
                draws += 1

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Eval | M1:{model1_wins} M2:{model2_wins} D:{draws} E:{errors}"
                )

        if progress_bar:
            progress_bar.close()

        total_played = model1_wins + model2_wins + draws
        model1_win_rate = model1_wins / total_played if total_played > 0 else 0
        model2_win_rate = model2_wins / total_played if total_played > 0 else 0
        draw_rate = draws / total_played if total_played > 0 else 0

        if verbose:
            print("\n--- Evaluation Summary ---")
            print(f"Total games finished: {total_played} (Errors: {errors})")
            print(f"Model1 Wins: {model1_wins} ({model1_win_rate:.2%})")
            print(f"Model2 Wins: {model2_wins} ({model2_win_rate:.2%})")
            print(f"Draws: {draws} ({draw_rate:.2%})")

            # Determine if model1 is better (e.g., win rate > 55%)
            win_threshold = 0.55 # Can be adjusted
            is_better = model1_win_rate > win_threshold
            verdict = "BETTER" if is_better else ("WORSE" if model2_win_rate > win_threshold else "UNCLEAR")
            print(f"Verdict (Win Threshold {win_threshold*100:.0f}%): Model1 is {verdict} than Model2")
            print("------------------------")

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_games_attempted": total_games,
            "total_games_finished": total_played,
            "errors": errors,
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "draws": draws,
            "model1_win_rate": model1_win_rate,
            "is_model1_better": is_better,
            "games_data": games_data # Contains detailed game info
        }

        self.save_results(results)
        return results

    def save_results(self, results):
        """
        Save evaluation results to a JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            if verbose: print(f"Evaluation results saved to {filepath}")
        except Exception as e:
            print(f"Error saving evaluation results to {filepath}: {e}")

# Removed index_to_move method as it's now imported from utils

def main(args):
    """ Main function to run evaluation from command line. """
    # Load challenger model (model1)
    print(f"Loading challenger model (Model 1) from: {args.challenger_model}")
    challenger_model = AlphaZeroChessNet().to(device)
    try:
        challenger_model.load_state_dict(torch.load(args.challenger_model, map_location=device))
    except Exception as e:
         print(f"Error loading challenger model: {e}")
         return

    # Load current best model (model2)
    print(f"Loading current best model (Model 2) from: {args.current_best_model}")
    current_best_model = AlphaZeroChessNet().to(device)
    try:
        current_best_model.load_state_dict(torch.load(args.current_best_model, map_location=device))
    except Exception as e:
         print(f"Error loading current best model: {e}")
         return

    print("Initializing Evaluator...")
    evaluator = Evaluator(
        model1=challenger_model,
        model2=current_best_model,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        max_moves=args.max_moves,
        results_dir=args.results_dir,
        swap_sides=not args.no_swap # swap_sides is True if --no-swap is NOT provided
    )

    print(f"Starting evaluation: {args.num_games} games, {args.num_simulations} sims/move...")
    results = evaluator.evaluate(verbose=True)

    # Optional: print detailed results or summary again
    # print(json.dumps(results, indent=4))
    print("Evaluation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero Chess Models")
    parser.add_argument("challenger_model", type=str, help="Path to the challenger model (.pth file)")
    parser.add_argument("current_best_model", type=str, help="Path to the current best model (.pth file)")
    parser.add_argument("--num_games", type=int, default=40, help="Number of games to play for evaluation")
    parser.add_argument("--num_simulations", type=int, default=400, help="Number of MCTS simulations per move")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for move selection (0.0 for deterministic)")
    parser.add_argument("--max_moves", type=int, default=512, help="Maximum moves per game before draw")
    parser.add_argument("--results_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--no-swap", action="store_true", help="Disable swapping sides (model1 always plays white)")
    # Add verbosity control?

    args = parser.parse_args()
    main(args) 