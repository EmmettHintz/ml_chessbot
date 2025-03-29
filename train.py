import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

from model import AlphaZeroChessNet, device
from utils import board_to_features
from self_play import preprocess_example


class ChessDataset(Dataset):
    """
    Dataset for training the AlphaZero Chess model.
    """
    def __init__(self, examples):
        """
        Initialize the dataset with a list of examples.
        
        Args:
            examples: List of tuples (state_fen, policy, value).
        """
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            state_tensor: Processed board features.
            policy_tensor: MCTS policy probabilities.
            value_tensor: Game outcome.
        """
        state_fen, policy, value = self.examples[idx]
        
        # Convert FEN to feature representation
        state = board_to_features(state_fen)
        state_tensor = torch.FloatTensor(state)
        
        # Convert policy to tensor
        policy_tensor = torch.FloatTensor(policy)
        
        # Convert value to tensor
        value_tensor = torch.FloatTensor([value])
        
        return state_tensor, policy_tensor, value_tensor


class Trainer:
    """
    Manages the training process for the AlphaZero model.
    """
    def __init__(
        self,
        model,
        optimizer=None,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=256,
        epochs=10,
        train_data_dir="training_data",
        models_dir="models",
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        log_interval=100,
        checkpoint_interval=1000,
        save_interval=1,
        max_examples=500000,  # Limit memory usage by using most recent examples
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The AlphaZeroChessNet model to train.
            optimizer: Optimizer (default: Adam).
            lr: Learning rate.
            weight_decay: L2 regularization parameter.
            batch_size: Training batch size.
            epochs: Number of training epochs per call.
            train_data_dir: Directory containing training data.
            models_dir: Directory to save model checkpoints.
            policy_loss_weight: Weight for policy loss.
            value_loss_weight: Weight for value loss.
            log_interval: Log metrics every N batches.
            checkpoint_interval: Save model every N batches.
            save_interval: Save model every N epochs.
            max_examples: Maximum number of examples to keep in memory.
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_data_dir = train_data_dir
        self.models_dir = models_dir
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.save_interval = save_interval
        self.max_examples = max_examples
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
            
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.iteration_counts = []
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss() # Can also use KL divergence
        self.value_criterion = nn.MSELoss()
        
        # Total training iterations completed
        self.iterations_completed = 0
        
    def load_examples(self, npz_files=None):
        """
        Load training examples from NPZ files.
        
        Args:
            npz_files: List of NPZ filenames, or None to load all files in train_data_dir.
            
        Returns:
            List of training examples.
        """
        all_examples = []
        
        # If no files specified, find all NPZ files in the data directory
        if npz_files is None:
            npz_files = glob.glob(os.path.join(self.train_data_dir, "*.npz"))
            
        if not npz_files:
            raise ValueError(f"No NPZ files found in {self.train_data_dir}")
            
        print(f"Loading {len(npz_files)} data files...")
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                states = data["states"]
                policies = data["policies"]
                results = data["results"]
                
                examples = [(s, p, r) for s, p, r in zip(states, policies, results)]
                all_examples.extend(examples)
                
                print(f"Loaded {len(examples)} examples from {npz_file}")
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                
        # Limit the number of examples to avoid memory issues
        if len(all_examples) > self.max_examples:
            print(f"Using the most recent {self.max_examples} examples (out of {len(all_examples)} total)")
            all_examples = all_examples[-self.max_examples:]
            
        return all_examples
    
    def train_epoch(self, dataloader, epoch):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: The DataLoader containing training examples.
            epoch: The epoch number (for logging).
            
        Returns:
            Mean policy loss, value loss, and total loss for this epoch.
        """
        self.model.train()
        
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, (states, policies, values) in enumerate(dataloader):
            # Move tensors to device
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_preds = self.model(states)
            
            # Calculate policy loss (cross-entropy)
            # Note: policy_logits are raw logits, policies are already probabilities
            # We can use CrossEntropyLoss with target probabilities (requires PyTorch >= 1.10.0)
            # Alternative: Use KL divergence loss between softmax(policy_logits) and policies
            policy_loss = -torch.sum(policies * torch.log_softmax(policy_logits, dim=1)) / policies.size(0)
            
            # Calculate value loss (MSE)
            value_loss = self.value_criterion(value_preds, values)
            
            # Total loss
            total_loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
            
            # Backward and optimize
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            
            # Increment counters
            batch_count += 1
            self.iterations_completed += 1
            
            # Log progress
            if (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Policy Loss: {policy_loss:.6f}, Value Loss: {value_loss:.6f}, "
                      f"Total Loss: {total_loss:.6f}, Time: {elapsed:.2f}s")
                start_time = time.time()
                
            # Save checkpoint periodically
            if self.iterations_completed % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.models_dir, 
                    f"checkpoint_iter_{self.iterations_completed}.pth"
                )
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
            # Store metrics for plotting
            if batch_idx % 10 == 0:  # Record every 10th batch to avoid too many points
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.total_losses.append(total_loss.item())
                self.iteration_counts.append(self.iterations_completed)
                
        # Compute mean losses
        mean_policy_loss = epoch_policy_loss / batch_count
        mean_value_loss = epoch_value_loss / batch_count
        mean_total_loss = epoch_total_loss / batch_count
        
        return mean_policy_loss, mean_value_loss, mean_total_loss
    
    def train(self, examples=None):
        """
        Train the model on a set of examples.
        
        Args:
            examples: List of examples to train on. If None, load from files.
            
        Returns:
            The trained model.
        """
        # Load examples if not provided
        if examples is None:
            examples = self.load_examples()
            
        if not examples:
            raise ValueError("No training examples provided")
            
        print(f"Training on {len(examples)} examples for {self.epochs} epochs")
        
        # Create dataset and dataloader
        dataset = ChessDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Train for specified number of epochs
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            # Train one epoch
            policy_loss, value_loss, total_loss = self.train_epoch(dataloader, epoch)
            
            # Step the scheduler
            self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{self.epochs} completed in {epoch_time:.2f}s")
            print(f"Mean Policy Loss: {policy_loss:.6f}")
            print(f"Mean Value Loss: {value_loss:.6f}")
            print(f"Mean Total Loss: {total_loss:.6f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save model every save_interval epochs
            if epoch % self.save_interval == 0:
                model_path = os.path.join(
                    self.models_dir,
                    f"model_epoch_{epoch}.pth"
                )
                self.save_model(model_path)
                print(f"Saved model to {model_path}")
                
        # Save final model
        final_model_path = os.path.join(self.models_dir, "model_final.pth")
        self.save_model(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.model
    
    def save_model(self, filepath):
        """
        Save the model weights to a file.
        
        Args:
            filepath: Path to save the model.
        """
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to the model weights file.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=device))
        
    def plot_training_curves(self):
        """
        Plot training loss curves and save to file.
        """
        if not self.iteration_counts:
            print("No training data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot policy loss
        plt.subplot(3, 1, 1)
        plt.plot(self.iteration_counts, self.policy_losses, 'b-')
        plt.title('Policy Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        
        # Plot value loss
        plt.subplot(3, 1, 2)
        plt.plot(self.iteration_counts, self.value_losses, 'r-')
        plt.title('Value Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        
        # Plot total loss
        plt.subplot(3, 1, 3)
        plt.plot(self.iteration_counts, self.total_losses, 'g-')
        plt.title('Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.models_dir, f"training_curves_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved training curves to {plot_path}")


def find_latest_data_files(data_dir, num_files=None):
    """
    Find the latest data files in the given directory.
    
    Args:
        data_dir: Directory containing NPZ data files.
        num_files: Number of latest files to return, or None for all.
        
    Returns:
        List of paths to the latest data files.
    """
    data_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # Sort by modification time (newest first)
    data_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if num_files is not None and num_files > 0:
        return data_files[:num_files]
    
    return data_files


def main(args):
    # Initialize the model
    print(f"Initializing model...")
    model = AlphaZeroChessNet(
        input_channels=18,  # Adjust based on feature representation
        num_filters=256,
        num_res_blocks=19,
        policy_output_size=1968
    ).to(device)
    
    # Load weights if specified
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("Training from scratch with random initialization")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_data_dir=args.data_dir,
        models_dir=args.models_dir,
        policy_loss_weight=args.policy_weight,
        value_loss_weight=args.value_weight,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        save_interval=args.save_interval,
        max_examples=args.max_examples
    )
    
    # Find data files to train on
    if args.latest_files:
        data_files = find_latest_data_files(args.data_dir, args.latest_files)
        print(f"Using {len(data_files)} latest data files")
    else:
        data_files = None  # Use all files in directory
    
    # Load examples and train
    examples = trainer.load_examples(data_files)
    trainer.train(examples)
    
    print("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file to resume training")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory containing training data")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--policy_weight", type=float, default=1.0, help="Weight for policy loss")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N batches")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N iterations")
    parser.add_argument("--save_interval", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--max_examples", type=int, default=500000, help="Maximum number of examples to use")
    parser.add_argument("--latest_files", type=int, default=None, help="Use only N latest data files")
    
    args = parser.parse_args()
    main(args) 