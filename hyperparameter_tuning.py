import os
import argparse
import json
import torch
import numpy as np
import random
import time
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis
import matplotlib.pyplot as plt
import logging
import pandas as pd
from tqdm import tqdm
import shutil
import traceback

from model import AlphaZeroChessNet, device
from mcts import MCTS
import chess # Use python-chess instead of Chessnut
from utils import get_legal_moves_mask, index_to_move

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Tuner:
    """
    Manages hyperparameter tuning for AlphaZero.
    """
    def __init__(
        self,
        tuning_type="mcts",  # One of: "mcts", "training", "model", "all"
        n_trials=100,
        n_jobs=1,
        study_name="alphazero_tuning",
        model_path=None,
        dataset_path=None,
        eval_positions_path="tuning_eval_positions.json",
        output_dir="tuning_results",
        test_model_path="models/best_model.pth",
        timeout=None,  # In seconds
    ):
        """
        Initialize the tuner.
        
        Args:
            tuning_type: Type of hyperparameters to tune.
            n_trials: Number of trials to run.
            n_jobs: Number of parallel jobs.
            study_name: Name of the study.
            model_path: Path to the model to use for tuning (if applicable).
            dataset_path: Path to the dataset to use for tuning (if applicable).
            eval_positions_path: Path to save/load evaluation positions.
            output_dir: Directory to save results.
            test_model_path: Path to baseline model for comparison.
            timeout: Maximum time (in seconds) for the entire study.
        """
        self.tuning_type = tuning_type
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.eval_positions_path = eval_positions_path
        self.output_dir = output_dir
        self.test_model_path = test_model_path
        self.timeout = timeout
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load or create evaluation positions
        self.eval_positions = self.load_or_create_eval_positions()
    
    def load_or_create_eval_positions(self, num_positions=50):
        """
        Load evaluation positions from file or create new ones.
        
        Args:
            num_positions: Number of positions to create if file doesn't exist.
            
        Returns:
            List of positions (FEN strings).
        """
        if os.path.exists(self.eval_positions_path):
            logger.info(f"Loading evaluation positions from {self.eval_positions_path}")
            with open(self.eval_positions_path, 'r') as f:
                positions = json.load(f)
            return positions
        
        logger.info(f"Creating {num_positions} evaluation positions")
        
        positions = []
        
        # Include the starting position
        positions.append("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Generate random positions by playing random moves from the starting position
        for i in range(num_positions - 1):
            board = chess.Board() # Use python-chess Board
            
            # Play a random number of moves (between 5 and 30)
            num_moves = random.randint(5, 30)
            
            for _ in range(num_moves):
                legal_moves = list(board.legal_moves)
                if not legal_moves or board.is_game_over():
                    break
                
                # Choose a random move
                move = random.choice(legal_moves)
                board.push(move)
            
            # Add the final position
            positions.append(board.fen())
        
        # Save positions to file
        with open(self.eval_positions_path, 'w') as f:
            json.dump(positions, f, indent=2)
        
        return positions
    
    def evaluate_mcts_settings(self, trial):
        """
        Evaluate MCTS hyperparameter settings on a set of positions.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Average evaluation score.
        """
        # Sample hyperparameters
        c_puct = trial.suggest_float("c_puct", 0.1, 5.0, log=True)
        num_simulations = trial.suggest_int("num_simulations", 50, 1000)
        
        # Create model
        model = AlphaZeroChessNet().to(device)
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()
        
        # Create MCTS with the sampled hyperparameters
        mcts = MCTS(model, c_puct=c_puct, num_simulations=num_simulations)
        
        # Baseline MCTS for comparison
        baseline_model = AlphaZeroChessNet().to(device)
        if self.test_model_path:
            baseline_model.load_state_dict(torch.load(self.test_model_path, map_location=device))
        baseline_model.eval()
        baseline_mcts = MCTS(baseline_model, c_puct=1.0, num_simulations=100)  # Default settings
        
        # Evaluate on positions
        scores = []
        
        for fen in tqdm(self.eval_positions[:10], desc="Evaluating positions", leave=False):
            # Get moves from both MCTS variants
            trial_probs, trial_value = mcts.get_action_probs(fen, temperature=0.1)
            baseline_probs, baseline_value = baseline_mcts.get_action_probs(fen, temperature=0.1)
            
            # Calculate JS Divergence between the two probability distributions
            # This measures how different the move selection is
            m = 0.5 * (trial_probs + baseline_probs)
            js_div = 0.5 * np.sum(trial_probs * np.log(trial_probs / m + 1e-10)) + 0.5 * np.sum(baseline_probs * np.log(baseline_probs / m + 1e-10))
            
            # Calculate agreement on top move
            trial_top_move = np.argmax(trial_probs)
            baseline_top_move = np.argmax(baseline_probs)
            move_agreement = 1.0 if trial_top_move == baseline_top_move else 0.0
            
            # Calculate score for this position
            # We want high agreement on best move, but some diversity in the probability distribution
            # Lower js_div means more similar distributions
            position_score = move_agreement - 0.1 * js_div
            
            scores.append(position_score)
        
        # Return the average score
        avg_score = np.mean(scores)
        
        # Print current trial results
        logger.info(f"Trial {trial.number}: c_puct={c_puct:.3f}, sims={num_simulations}, score={avg_score:.3f}")
        
        return avg_score
    
    def evaluate_training_settings(self, trial):
        """
        Evaluate training hyperparameter settings.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Validation loss.
        """
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        policy_weight = trial.suggest_float("policy_weight", 0.5, 5.0)
        value_weight = trial.suggest_float("value_weight", 0.5, 5.0)
        
        # Since full training is expensive, we'll use a small subset and few epochs
        try:
            # Import Trainer class without redefining it
            from train import Trainer, ChessDataset
            
            # Load a small subset of data
            data_files = [self.dataset_path] if self.dataset_path else None
            
            # Create model
            model = AlphaZeroChessNet().to(device)
            if self.model_path:
                model.load_state_dict(torch.load(self.model_path, map_location=device))
            
            # Create trainer
            trainer = Trainer(
                model=model,
                lr=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=1,  # Use just one epoch for quick evaluation
                policy_loss_weight=policy_weight,
                value_loss_weight=value_weight
            )
            
            # Load a small subset of examples
            examples = trainer.load_examples(data_files)
            random.shuffle(examples)
            examples = examples[:min(len(examples), 10000)]  # Limit to 10,000 examples
            
            # Split into train/validation
            split_idx = int(0.8 * len(examples))
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            # Create datasets
            train_dataset = ChessDataset(train_examples)
            val_dataset = ChessDataset(val_examples)
            
            # Train for a fraction of an epoch
            torch.cuda.empty_cache()
            trainer.train(train_examples)
            
            # Evaluate on validation set
            val_policy_loss, val_value_loss, _ = trainer.evaluate_dataset(val_dataset)
            
            # Calculate combined loss
            combined_loss = policy_weight * val_policy_loss + value_weight * val_value_loss
            
            # Print current trial results
            logger.info(f"Trial {trial.number}: lr={learning_rate:.6f}, batch={batch_size}, "
                        f"p_weight={policy_weight:.2f}, v_weight={value_weight:.2f}, "
                        f"val_loss={combined_loss:.6f}")
            
            return combined_loss
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            logger.error(traceback.format_exc())
            return float('inf')  # Return a high loss on error
    
    def evaluate_model_architecture(self, trial):
        """
        Evaluate model architecture hyperparameter settings.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Combined evaluation score.
        """
        # Sample hyperparameters
        num_filters = trial.suggest_categorical("num_filters", [64, 128, 192, 256])
        num_res_blocks = trial.suggest_int("num_res_blocks", 5, 20)
        
        try:
            # Create model with the sampled architecture
            model = AlphaZeroChessNet(
                num_filters=num_filters,
                num_res_blocks=num_res_blocks
            ).to(device)
            
            # Initialize with random weights or load existing weights if possible
            if self.model_path:
                try:
                    model.load_state_dict(torch.load(self.model_path, map_location=device))
                except:
                    logger.warning("Failed to load existing weights - using random initialization")
            
            model.eval()
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Create MCTS
            mcts = MCTS(model, c_puct=1.0, num_simulations=100)
            
            # Evaluate on a few positions
            eval_scores = []
            inference_times = []
            
            for fen in tqdm(self.eval_positions[:5], desc="Evaluating positions", leave=False):
                # Measure inference time
                start_time = time.time()
                
                # Get model predictions (without MCTS)
                features = board_to_features(fen)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy_logits, value_output = model(features_tensor)
                
                # Record inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Run a quick MCTS
                # We check if MCTS works with this model configuration
                try:
                    action_probs, value = mcts.get_action_probs(fen, temperature=0.1)
                    
                    # Check that we're producing valid outputs
                    valid_moves = np.where(action_probs > 0)[0]
                    if len(valid_moves) > 0:
                        eval_scores.append(1.0)  # Score of 1 for successful evaluation
                    else:
                        eval_scores.append(0.0)  # Score of 0 for unsuccessful evaluation
                except Exception as e:
                    logger.error(f"MCTS evaluation failed: {e}")
                    eval_scores.append(0.0)  # Score of 0 for failed evaluation
            
            # Calculate scores
            avg_eval_score = np.mean(eval_scores)
            avg_inference_time = np.mean(inference_times)
            
            # Calculate combined score
            # We want low inference time and successful evaluations
            # Parameter count penalty to avoid extremely large models
            param_penalty = np.log(num_params) / 20.0  # Penalty increases with parameter count
            
            combined_score = avg_eval_score - 5.0 * avg_inference_time - param_penalty
            
            # Print current trial results
            logger.info(f"Trial {trial.number}: filters={num_filters}, res_blocks={num_res_blocks}, "
                        f"params={num_params:,}, avg_time={avg_inference_time:.4f}s, score={combined_score:.4f}")
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            logger.error(traceback.format_exc())
            return -float('inf')  # Return a very low score on error
    
    def run_optimization(self):
        """
        Run the hyperparameter optimization study.
        
        Returns:
            The Optuna study object.
        """
        # Create Optuna study with TPE sampler
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # We want to maximize the score/minimize loss
            sampler=TPESampler(seed=42)
        )
        
        # Set the appropriate objective function
        if self.tuning_type == "mcts":
            objective = self.evaluate_mcts_settings
        elif self.tuning_type == "training":
            objective = self.evaluate_training_settings
        elif self.tuning_type == "model":
            objective = self.evaluate_model_architecture
        else:  # "all" or any other value
            raise ValueError(f"Unsupported tuning type: {self.tuning_type}")
        
        # Run the optimization
        logger.info(f"Starting {self.tuning_type} hyperparameter optimization for {self.n_trials} trials")
        start_time = time.time()
        
        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs, timeout=self.timeout)
        except KeyboardInterrupt:
            logger.info("Optimization stopped manually.")
        
        # Calculate time taken
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.1f} seconds")
        
        # Print best parameters and score
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value:.6f}")
        
        # Save results
        self.save_results(study)
        
        return study
    
    def save_results(self, study):
        """
        Save optimization results.
        
        Args:
            study: Optuna study object.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectory for this tuning run
        run_dir = os.path.join(self.output_dir, f"{self.tuning_type}_tuning_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save study statistics as JSON
        stats = {
            "tuning_type": self.tuning_type,
            "n_trials": self.n_trials,
            "timestamp": timestamp,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_completed_trials": len(study.trials)
        }
        
        with open(os.path.join(run_dir, "study_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save all trial data as CSV
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(run_dir, "trials.csv"), index=False)
        
        # Save visualizations
        if len(study.trials) > 1:
            try:
                # Parameter importances
                param_importances = vis.plot_param_importances(study)
                param_importances.write_image(os.path.join(run_dir, "param_importances.png"))
                
                # Optimization history
                opt_history = vis.plot_optimization_history(study)
                opt_history.write_image(os.path.join(run_dir, "optimization_history.png"))
                
                # Slice plot for each parameter
                for param_name in study.best_params.keys():
                    slice_plot = vis.plot_slice(study, params=[param_name])
                    slice_plot.write_image(os.path.join(run_dir, f"slice_{param_name}.png"))
                
                # Contour plots for pairs of important parameters
                param_pairs = []
                for i, p1 in enumerate(study.best_params.keys()):
                    for p2 in list(study.best_params.keys())[i+1:]:
                        param_pairs.append((p1, p2))
                
                for p1, p2 in param_pairs[:5]:  # Limit to first 5 pairs
                    contour_plot = vis.plot_contour(study, params=[p1, p2])
                    contour_plot.write_image(os.path.join(run_dir, f"contour_{p1}_{p2}.png"))
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
        
        logger.info(f"Results saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Hyperparameter Tuning")
    
    parser.add_argument("--type", type=str, choices=["mcts", "training", "model"], required=True,
                        help="Type of hyperparameters to tune")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--model", type=str, default=None, help="Path to model to tune")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset for training tuning")
    parser.add_argument("--eval_positions", type=str, default="tuning_eval_positions.json",
                        help="Path to positions for evaluation")
    parser.add_argument("--output_dir", type=str, default="tuning_results", help="Directory to save results")
    parser.add_argument("--test_model", type=str, default="models/best_model.pth",
                        help="Path to baseline model for comparison")
    parser.add_argument("--timeout", type=int, default=None, help="Maximum time in seconds")
    
    args = parser.parse_args()
    
    # Check if model exists if provided
    if args.model and not os.path.exists(args.model):
        parser.error(f"Model file not found: {args.model}")
    
    # Create tuner
    tuner = Tuner(
        tuning_type=args.type,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        model_path=args.model,
        dataset_path=args.dataset,
        eval_positions_path=args.eval_positions,
        output_dir=args.output_dir,
        test_model_path=args.test_model,
        timeout=args.timeout
    )
    
    # Run optimization
    study = tuner.run_optimization()
    
    print(f"\nBest {args.type} hyperparameters found:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"Best score: {study.best_value:.6f}")


if __name__ == "__main__":
    # Import board_to_features here to ensure it's available when needed
    from utils import board_to_features
    main() 