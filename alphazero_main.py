import os
import argparse
import torch
import shutil
import json
import time
from datetime import datetime
import logging
from pathlib import Path

# Our AlphaZero components
from model import AlphaZeroChessNet, device
from self_play import SelfPlay
from train import Trainer
from evaluate import Evaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alphazero_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaZeroTrainingPipeline:
    """
    Manages the complete AlphaZero training pipeline:
    1. Self-play data generation
    2. Neural network training
    3. Model evaluation
    4. Iteration
    """
    def __init__(
        self,
        # Directories
        models_dir="models",
        best_model_path="models/best_model.pth",
        training_data_dir="training_data",
        evaluation_results_dir="evaluation_results",
        
        # Model parameters
        input_channels=18,
        num_filters=256,
        num_res_blocks=19,
        policy_output_size=1968,
        
        # Self-play parameters
        self_play_games=100,
        self_play_simulations=100,
        self_play_temperature_threshold=15,
        self_play_max_moves=512,
        
        # Training parameters
        training_epochs=10,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        
        # Evaluation parameters
        eval_games=40,
        eval_simulations=400,
        win_threshold=0.55,
        
        # Iteration parameters
        iterations=100,
        checkpoint_frequency=1,
    ):
        """
        Initialize the training pipeline.
        
        Args:
            models_dir: Directory to store model checkpoints
            best_model_path: Path to store the current best model
            training_data_dir: Directory to store self-play data
            evaluation_results_dir: Directory to store evaluation results
            
            input_channels: Number of input channels for the model
            num_filters: Number of filters in convolutional layers
            num_res_blocks: Number of residual blocks in the model
            policy_output_size: Dimensionality of policy output
            
            self_play_games: Number of games to play during self-play
            self_play_simulations: Number of MCTS simulations per move during self-play
            self_play_temperature_threshold: Move threshold for temperature annealing
            self_play_max_moves: Maximum number of moves per game
            
            training_epochs: Number of epochs to train the model
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            
            eval_games: Number of games to play during evaluation
            eval_simulations: Number of MCTS simulations per move during evaluation
            win_threshold: Win rate threshold to determine if a model is better
            
            iterations: Number of training iterations to run
            checkpoint_frequency: How often to save iteration checkpoints
        """
        # Store parameters
        self.models_dir = models_dir
        self.best_model_path = best_model_path
        self.training_data_dir = training_data_dir
        self.evaluation_results_dir = evaluation_results_dir
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_output_size = policy_output_size
        
        self.self_play_games = self_play_games
        self.self_play_simulations = self_play_simulations
        self.self_play_temperature_threshold = self_play_temperature_threshold
        self.self_play_max_moves = self_play_max_moves
        
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.eval_games = eval_games
        self.eval_simulations = eval_simulations
        self.win_threshold = win_threshold
        
        self.iterations = iterations
        self.checkpoint_frequency = checkpoint_frequency
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(training_data_dir, exist_ok=True)
        os.makedirs(evaluation_results_dir, exist_ok=True)
        
        # Initialize the model
        self.model = self._create_model()
        
        # Training metrics to track progress
        self.training_history = {
            "iteration": [],
            "self_play_games": [],
            "training_policy_loss": [],
            "training_value_loss": [],
            "evaluation_win_rate": [],
            "time_elapsed": []
        }
        
    def _create_model(self):
        """Create a new instance of the AlphaZero model."""
        return AlphaZeroChessNet(
            input_channels=self.input_channels,
            num_filters=self.num_filters,
            num_res_blocks=self.num_res_blocks,
            policy_output_size=self.policy_output_size
        ).to(device)
        
    def initialize_with_random_weights(self):
        """Initialize the best model with random weights."""
        if not os.path.exists(self.best_model_path):
            logger.info(f"Initializing best model with random weights at {self.best_model_path}")
            # Save the randomly initialized model
            torch.save(self.model.state_dict(), self.best_model_path)
        else:
            logger.info(f"Best model already exists at {self.best_model_path}")
            
    def run_self_play(self, iteration):
        """
        Run the self-play phase using the best model.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Number of games played
        """
        # Load the best model
        best_model = self._create_model()
        best_model.load_state_dict(torch.load(self.best_model_path, map_location=device))
        best_model.eval()
        
        logger.info(f"Iteration {iteration}: Starting self-play phase with {self.self_play_games} games")
        
        # Create self-play manager
        self_play = SelfPlay(
            model=best_model,
            num_games=self.self_play_games,
            num_simulations=self.self_play_simulations,
            c_puct=1.0,
            temperature_threshold=self.self_play_temperature_threshold,
            max_moves=self.self_play_max_moves,
            data_dir=self.training_data_dir
        )
        
        # Generate self-play data
        examples = self_play.generate_data(verbose=True)
        
        logger.info(f"Iteration {iteration}: Self-play completed. Generated {len(examples)} training examples.")
        
        return len(examples)
    
    def run_training(self, iteration):
        """
        Train the neural network on data from self-play.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dictionary with training metrics
        """
        # Create a new model instance
        training_model = self._create_model()
        
        # Load the best model weights as starting point
        training_model.load_state_dict(torch.load(self.best_model_path, map_location=device))
        
        logger.info(f"Iteration {iteration}: Starting training phase")
        
        # Create trainer
        trainer = Trainer(
            model=training_model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            epochs=self.training_epochs,
            train_data_dir=self.training_data_dir,
            models_dir=self.models_dir
        )
        
        # Train on most recent data
        trainer.train()
        
        # Save the trained model
        trained_model_path = os.path.join(self.models_dir, f"model_iteration_{iteration}.pth")
        torch.save(training_model.state_dict(), trained_model_path)
        
        logger.info(f"Iteration {iteration}: Training completed. Model saved to {trained_model_path}")
        
        # Return the training statistics
        return {
            "policy_loss": trainer.policy_losses[-1] if trainer.policy_losses else float('nan'),
            "value_loss": trainer.value_losses[-1] if trainer.value_losses else float('nan'),
            "model_path": trained_model_path
        }
    
    def run_evaluation(self, iteration, trained_model_path):
        """
        Evaluate the trained model against the current best model.
        
        Args:
            iteration: Current iteration number
            trained_model_path: Path to the trained model
            
        Returns:
            True if the trained model is better, False otherwise
        """
        # Load the newly trained model (challenger)
        challenger_model = self._create_model()
        challenger_model.load_state_dict(torch.load(trained_model_path, map_location=device))
        challenger_model.eval()
        
        # Load the current best model
        best_model = self._create_model()
        best_model.load_state_dict(torch.load(self.best_model_path, map_location=device))
        best_model.eval()
        
        logger.info(f"Iteration {iteration}: Starting evaluation phase with {self.eval_games} games")
        
        # Create evaluator
        evaluator = Evaluator(
            model1=challenger_model,
            model2=best_model,
            num_games=self.eval_games,
            num_simulations=self.eval_simulations,
            c_puct=1.0,
            temperature=0.1,  # Lower temperature for evaluation
            max_moves=self.self_play_max_moves,
            results_dir=self.evaluation_results_dir,
            swap_sides=True
        )
        
        # Run evaluation
        results = evaluator.evaluate(verbose=True)
        
        # Determine if the challenger is better
        is_better = results["model1_win_rate"] > self.win_threshold
        
        if is_better:
            logger.info(f"Iteration {iteration}: New model is better! Win rate: {results['model1_win_rate']:.2%}")
            # Update the best model
            shutil.copy2(trained_model_path, self.best_model_path)
            logger.info(f"Iteration {iteration}: Best model updated to {trained_model_path}")
        else:
            logger.info(f"Iteration {iteration}: Current best model is still better. Challenger win rate: {results['model1_win_rate']:.2%}")
        
        return {
            "win_rate": results["model1_win_rate"],
            "is_better": is_better
        }
    
    def run_iteration(self, iteration):
        """
        Run a complete iteration of the AlphaZero training pipeline.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dictionary with iteration metrics
        """
        iteration_start_time = time.time()
        
        logger.info(f"Starting iteration {iteration}/{self.iterations}")
        
        # Phase 1: Self-play
        num_examples = self.run_self_play(iteration)
        
        # Phase 2: Training
        training_stats = self.run_training(iteration)
        
        # Phase 3: Evaluation
        evaluation_stats = self.run_evaluation(iteration, training_stats["model_path"])
        
        # Record metrics
        iteration_time = time.time() - iteration_start_time
        
        self.training_history["iteration"].append(iteration)
        self.training_history["self_play_games"].append(num_examples)
        self.training_history["training_policy_loss"].append(training_stats["policy_loss"])
        self.training_history["training_value_loss"].append(training_stats["value_loss"])
        self.training_history["evaluation_win_rate"].append(evaluation_stats["win_rate"])
        self.training_history["time_elapsed"].append(iteration_time)
        
        # Save training history
        self.save_training_history()
        
        # Log iteration summary
        logger.info(f"Iteration {iteration} completed in {iteration_time:.1f} seconds")
        logger.info(f"  Self-play examples: {num_examples}")
        logger.info(f"  Training policy loss: {training_stats['policy_loss']:.6f}")
        logger.info(f"  Training value loss: {training_stats['value_loss']:.6f}")
        logger.info(f"  Evaluation win rate: {evaluation_stats['win_rate']:.2%}")
        logger.info(f"  Model improved: {evaluation_stats['is_better']}")
        
        return {
            "num_examples": num_examples,
            "policy_loss": training_stats["policy_loss"],
            "value_loss": training_stats["value_loss"],
            "win_rate": evaluation_stats["win_rate"],
            "is_better": evaluation_stats["is_better"],
            "time": iteration_time
        }
    
    def save_training_history(self):
        """Save the training history to a JSON file."""
        history_path = os.path.join(self.models_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def run_pipeline(self):
        """Run the complete AlphaZero training pipeline for the specified number of iterations."""
        pipeline_start_time = time.time()
        
        logger.info(f"Starting AlphaZero training pipeline with {self.iterations} iterations")
        
        # Make sure the best model exists (initialize with random weights if needed)
        self.initialize_with_random_weights()
        
        # Run iterations
        for iteration in range(1, self.iterations + 1):
            self.run_iteration(iteration)
            
            # Save checkpoint of the pipeline state
            if iteration % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.models_dir, f"checkpoint_iteration_{iteration}.json")
                with open(checkpoint_path, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
                logger.info(f"Saved pipeline checkpoint to {checkpoint_path}")
        
        # Calculate total time
        total_time = time.time() - pipeline_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"AlphaZero training pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Final best model saved to {self.best_model_path}")
        
        return self.training_history


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training Pipeline")
    
    # Directory parameters
    parser.add_argument("--models-dir", type=str, default="models", dest="models_dir", help="Directory for model checkpoints")
    parser.add_argument("--best-model", type=str, default="models/best_model.pth", dest="best_model_path", help="Path to store best model")
    parser.add_argument("--data-dir", type=str, default="training_data", dest="training_data_dir", help="Directory for self-play data")
    parser.add_argument("--eval-dir", type=str, default="evaluation_results", dest="evaluation_results_dir", help="Directory for evaluation results")
    
    # Model parameters
    parser.add_argument("--input-channels", type=int, default=18, dest="input_channels", help="Number of input channels for the model")
    parser.add_argument("--filters", type=int, default=256, dest="num_filters", help="Number of filters in convolutional layers")
    parser.add_argument("--res-blocks", type=int, default=19, dest="num_res_blocks", help="Number of residual blocks in the model")
    
    # Self-play parameters
    parser.add_argument("--self-play-games", type=int, default=100, dest="self_play_games", help="Number of self-play games per iteration")
    parser.add_argument("--self-play-sims", type=int, default=100, dest="self_play_simulations", help="MCTS simulations per move in self-play")
    
    # Training parameters
    parser.add_argument("--train-epochs", type=int, default=10, dest="training_epochs", help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size", help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, dest="learning_rate", help="Learning rate")
    
    # Evaluation parameters
    parser.add_argument("--eval-games", type=int, default=40, dest="eval_games", help="Number of evaluation games")
    parser.add_argument("--eval-sims", type=int, default=400, dest="eval_simulations", help="MCTS simulations per move in evaluation")
    parser.add_argument("--win-threshold", type=float, default=0.55, dest="win_threshold", help="Win rate threshold to update best model")
    
    # Pipeline parameters
    parser.add_argument("--iterations", type=int, default=100, dest="iterations", help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=1, dest="checkpoint_frequency", help="How often to save checkpoints")

    args = parser.parse_args()

    # Initialize the pipeline with parsed arguments
    pipeline = AlphaZeroTrainingPipeline(
        models_dir=args.models_dir,
        best_model_path=args.best_model_path,
        training_data_dir=args.training_data_dir,
        evaluation_results_dir=args.evaluation_results_dir,
        
        input_channels=args.input_channels,
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
        # policy_output_size might need to be set based on chess rules or model architecture
        
        self_play_games=args.self_play_games,
        self_play_simulations=args.self_play_simulations,
        # self_play_temperature_threshold might need default or arg
        # self_play_max_moves might need default or arg
        
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # weight_decay might need default or arg
        
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        win_threshold=args.win_threshold,
        
        iterations=args.iterations,
        checkpoint_frequency=args.checkpoint_frequency
    )

    # Run the training pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 