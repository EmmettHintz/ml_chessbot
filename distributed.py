import os
import sys
import time
import json
import logging
import socket
import threading
import queue
import multiprocessing
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def setup_worker(rank, world_size, master_addr, master_port, backend="nccl"):
    """
    Initialize the distributed environment for a worker.
    
    Args:
        rank: Unique identifier for this process
        world_size: Total number of processes
        master_addr: IP address of the master node
        master_port: Port for communication
        backend: PyTorch distributed backend
    """
    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)  # Set different seed for each process
    
    logger.info(f"Worker {rank}/{world_size} initialized on {socket.gethostname()}")


def cleanup_worker():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


class SelfPlayWorker:
    """
    Worker that generates self-play games for training.
    """
    def __init__(self, rank, world_size, model, config):
        """
        Initialize a self-play worker.
        
        Args:
            rank: Global rank of this worker
            world_size: Total number of workers
            model: Neural network model (loaded on this worker)
            config: Configuration dictionary
        """
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.config = config
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" 
                                  if torch.cuda.is_available() else "cpu")
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        
        # If using distributed training, wrap model in DDP
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device.index] 
                             if self.device.type == "cuda" else None)
        
        # Import mcts and other modules here to avoid circular imports
        from mcts import MCTS
        self.mcts = MCTS(self.model, config, device=self.device)
        
        # Create a queue for generated games
        self.game_queue = queue.Queue()
    
    def generate_games(self, num_games):
        """
        Generate self-play games.
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            List of game data dictionaries
        """
        from self_play import play_game
        
        games = []
        for i in range(num_games):
            logger.info(f"Worker {self.rank}: Starting game {i+1}/{num_games}")
            game_data = play_game(self.model, self.mcts, self.config, 
                                 device=self.device)
            games.append(game_data)
            logger.info(f"Worker {self.rank}: Completed game {i+1}/{num_games}")
        
        return games
    
    def run(self, num_games_per_worker):
        """
        Run the self-play worker to generate games.
        
        Args:
            num_games_per_worker: Number of games to generate per worker
        """
        games = self.generate_games(num_games_per_worker)
        
        # Here we could push the games to a central repository or storage
        return games


class DistributedTrainer:
    """
    Handles distributed training across multiple machines.
    """
    def __init__(self, config):
        """
        Initialize a distributed trainer.
        
        Args:
            config: Configuration dictionary with distributed settings
        """
        self.config = config
        self.world_size = config.get("world_size", 1)
        self.master_addr = config.get("master_addr", "localhost")
        self.master_port = config.get("master_port", 29500)
        self.backend = config.get("backend", "nccl" if torch.cuda.is_available() else "gloo")
        self.is_master = True  # Will be set properly when distributed training starts
    
    def _load_model(self):
        """
        Load the neural network model.
        
        Returns:
            Loaded model
        """
        from model import AlphaZeroChessNet
        
        # Load model configuration
        model_config = self.config.get("model", {})
        model = AlphaZeroChessNet(
            num_res_blocks=model_config.get("num_res_blocks", 19),
            num_hidden=model_config.get("num_hidden", 256)
        )
        
        # Load weights if available
        weights_path = self.config.get("weights_path", None)
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def _init_worker(self, rank, size):
        """
        Initialize a worker for distributed training.
        
        Args:
            rank: Global rank of this worker
            size: Total number of workers
        """
        # Set up the distributed environment
        setup_worker(rank, size, self.master_addr, self.master_port, self.backend)
        
        # Determine if this is the master process
        self.is_master = (rank == 0)
        
        # Load the model
        model = self._load_model()
        
        # Create a self-play worker
        worker = SelfPlayWorker(rank, size, model, self.config)
        
        return worker
    
    @record
    def _worker_process(self, rank, size, num_games, result_queue=None):
        """
        Function to run on each worker process.
        
        Args:
            rank: Global rank of this worker
            size: Total number of workers
            num_games: Total number of games to generate
            result_queue: Queue to store results (if None, results are saved to disk)
        """
        try:
            # Initialize worker
            worker = self._init_worker(rank, size)
            
            # Calculate number of games per worker
            num_games_per_worker = (num_games + size - 1) // size
            if rank == size - 1:
                # Last worker may need to generate fewer games
                remaining = num_games - (size - 1) * num_games_per_worker
                num_games_per_worker = max(0, remaining)
            
            # Generate games
            games = worker.run(num_games_per_worker)
            
            # Save games to queue or disk
            if result_queue is not None:
                result_queue.put((rank, games))
            else:
                # Save to disk with rank to avoid conflicts
                output_dir = self.config.get("output_dir", "data/self_play")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"games_worker_{rank}.json")
                with open(output_path, "w") as f:
                    json.dump(games, f)
                logger.info(f"Worker {rank}: Saved {len(games)} games to {output_path}")
            
            # Clean up
            cleanup_worker()
            
            return games
        except Exception as e:
            logger.error(f"Error in worker {rank}: {str(e)}")
            raise
    
    def generate_games(self, num_games, output_dir=None):
        """
        Generate self-play games across multiple processes.
        
        Args:
            num_games: Total number of games to generate
            output_dir: Directory to save games (if None, uses config value)
            
        Returns:
            List of all generated games
        """
        if output_dir:
            self.config["output_dir"] = output_dir
        
        # Create a multiprocessing queue for results
        result_queue = multiprocessing.Queue()
        
        # Start worker processes
        processes = []
        for rank in range(self.world_size):
            p = multiprocessing.Process(
                target=self._worker_process,
                args=(rank, self.world_size, num_games, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        all_games = []
        for _ in range(self.world_size):
            try:
                rank, games = result_queue.get(timeout=3600)  # 1 hour timeout
                all_games.extend(games)
                logger.info(f"Received {len(games)} games from worker {rank}")
            except queue.Empty:
                logger.warning("Timeout waiting for games from a worker")
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        logger.info(f"Generated a total of {len(all_games)} games")
        return all_games
    
    def train_distributed(self, num_epochs, batch_size, lr=0.001):
        """
        Train the model in a distributed setting.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size per GPU
            lr: Learning rate
            
        Returns:
            Trained model state dict
        """
        # This would be implemented to use DDP for distributed training
        # For simplicity, we'll outline the key steps
        
        # 1. Each process would initialize distributed environment
        # 2. Create model and wrap with DDP
        # 3. Load the dataset (each process loads its own shard)
        # 4. Train the model using standard PyTorch training loop
        # 5. Master process saves the final model
        
        logger.info("Distributed training not fully implemented in this demo")
        logger.info(f"Would train for {num_epochs} epochs with batch size {batch_size} and lr {lr}")
        
        # For demonstration, we'll just return the current model
        model = self._load_model()
        return model.state_dict()


class RemoteWorker:
    """
    Client-side implementation for connecting to a remote server.
    Useful for distributed computation across multiple machines.
    """
    def __init__(self, server_addr, server_port, worker_id=None):
        """
        Initialize a remote worker.
        
        Args:
            server_addr: IP address of the server
            server_port: Port for communication
            worker_id: Unique identifier for this worker (defaults to hostname)
        """
        self.server_addr = server_addr
        self.server_port = server_port
        self.worker_id = worker_id or socket.gethostname()
        self.socket = None
        self.running = False
        self.current_task = None
    
    def connect(self):
        """
        Connect to the server.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_addr, self.server_port))
            # Send worker ID
            self.socket.sendall(json.dumps({"worker_id": self.worker_id}).encode())
            response = json.loads(self.socket.recv(1024).decode())
            if response.get("status") == "connected":
                logger.info(f"Connected to server at {self.server_addr}:{self.server_port}")
                return True
            else:
                logger.error(f"Connection rejected: {response.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Disconnect from the server.
        """
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.running = False
    
    def _process_task(self, task):
        """
        Process a task received from the server.
        
        Args:
            task: Task data from the server
            
        Returns:
            Task result
        """
        task_type = task.get("type")
        task_id = task.get("id")
        
        if task_type == "self_play":
            # Task to generate self-play games
            num_games = task.get("num_games", 1)
            config = task.get("config", {})
            
            # Load model weights
            model_data = task.get("model_weights")
            model_path = "temp_model.pt"
            with open(model_path, "wb") as f:
                f.write(model_data)
            
            # Create model and generate games
            from model import AlphaZeroChessNet
            model = AlphaZeroChessNet()
            model.load_state_dict(torch.load(model_path))
            
            # Create self-play worker and generate games
            worker = SelfPlayWorker(0, 1, model, config)
            games = worker.generate_games(num_games)
            
            # Clean up
            os.remove(model_path)
            
            return {
                "type": "result",
                "task_id": task_id,
                "result": games
            }
        
        elif task_type == "evaluate":
            # Task to evaluate model performance
            num_games = task.get("num_games", 1)
            config = task.get("config", {})
            
            # Load model weights
            model1_data = task.get("model1_weights")
            model2_data = task.get("model2_weights")
            
            model1_path = "temp_model1.pt"
            model2_path = "temp_model2.pt"
            
            with open(model1_path, "wb") as f:
                f.write(model1_data)
            
            with open(model2_path, "wb") as f:
                f.write(model2_data)
            
            # Create models and evaluate
            from model import AlphaZeroChessNet
            from evaluate import Evaluator
            
            model1 = AlphaZeroChessNet()
            model2 = AlphaZeroChessNet()
            
            model1.load_state_dict(torch.load(model1_path))
            model2.load_state_dict(torch.load(model2_path))
            
            evaluator = Evaluator(
                model1, model2, num_games=num_games, config=config)
            result = evaluator.evaluate()
            
            # Clean up
            os.remove(model1_path)
            os.remove(model2_path)
            
            return {
                "type": "result",
                "task_id": task_id,
                "result": result
            }
        
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {
                "type": "error",
                "task_id": task_id,
                "error": f"Unknown task type: {task_type}"
            }
    
    def run(self):
        """
        Run the worker, processing tasks from the server.
        """
        if not self.socket:
            if not self.connect():
                return
        
        self.running = True
        
        while self.running:
            try:
                # Request a task
                self.socket.sendall(json.dumps({"type": "request_task"}).encode())
                
                # Receive task
                data = self.socket.recv(4096)
                if not data:
                    logger.warning("Server disconnected")
                    break
                
                task = json.loads(data.decode())
                
                if task.get("type") == "no_task":
                    # No task available, wait and try again
                    time.sleep(5)
                    continue
                
                if task.get("type") == "shutdown":
                    # Server requested shutdown
                    logger.info("Shutdown requested by server")
                    break
                
                # Process the task
                logger.info(f"Received task: {task.get('id')} ({task.get('type')})")
                self.current_task = task
                
                # Process task and send result
                result = self._process_task(task)
                self.socket.sendall(json.dumps(result).encode())
                self.current_task = None
                
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                time.sleep(5)  # Wait before trying again
        
        self.disconnect()


class TaskServer:
    """
    Server that distributes tasks to remote workers.
    """
    def __init__(self, port=29501, max_workers=16):
        """
        Initialize a task server.
        
        Args:
            port: Port to listen for connections
            max_workers: Maximum number of workers to accept
        """
        self.port = port
        self.max_workers = max_workers
        self.workers = {}  # worker_id -> connection
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.server_socket = None
    
    def start(self):
        """
        Start the task server.
        """
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", self.port))
        self.server_socket.listen(self.max_workers)
        
        self.running = True
        
        # Start connection handler thread
        connection_thread = threading.Thread(target=self._handle_connections)
        connection_thread.daemon = True
        connection_thread.start()
        
        logger.info(f"Task server started on port {self.port}")
    
    def stop(self):
        """
        Stop the task server.
        """
        self.running = False
        
        # Send shutdown signal to all workers
        for worker_id, conn in self.workers.items():
            try:
                conn.sendall(json.dumps({"type": "shutdown"}).encode())
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        logger.info("Task server stopped")
    
    def _handle_connections(self):
        """
        Handle incoming worker connections.
        """
        while self.running:
            try:
                # Accept new connection
                client_socket, addr = self.server_socket.accept()
                client_socket.settimeout(600)  # 10 minute timeout
                
                # Start a thread to handle this worker
                worker_thread = threading.Thread(
                    target=self._handle_worker,
                    args=(client_socket, addr)
                )
                worker_thread.daemon = True
                worker_thread.start()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {str(e)}")
    
    def _handle_worker(self, client_socket, addr):
        """
        Handle communication with a worker.
        
        Args:
            client_socket: Socket connection to the worker
            addr: Address of the worker
        """
        worker_id = None
        
        try:
            # Receive worker identification
            data = client_socket.recv(1024)
            if not data:
                return
            
            worker_info = json.loads(data.decode())
            worker_id = worker_info.get("worker_id")
            
            if not worker_id:
                client_socket.sendall(json.dumps({
                    "status": "error",
                    "message": "No worker ID provided"
                }).encode())
                return
            
            if len(self.workers) >= self.max_workers:
                client_socket.sendall(json.dumps({
                    "status": "error",
                    "message": "Maximum number of workers reached"
                }).encode())
                return
            
            # Accept the worker
            self.workers[worker_id] = client_socket
            client_socket.sendall(json.dumps({
                "status": "connected",
                "message": "Worker connected successfully"
            }).encode())
            
            logger.info(f"Worker {worker_id} connected from {addr}")
            
            # Handle worker requests
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                request = json.loads(data.decode())
                
                if request.get("type") == "request_task":
                    # Worker is requesting a task
                    try:
                        task = self.task_queue.get(block=False)
                        client_socket.sendall(json.dumps(task).encode())
                    except queue.Empty:
                        # No task available
                        client_socket.sendall(json.dumps({
                            "type": "no_task"
                        }).encode())
                
                elif request.get("type") == "result":
                    # Worker is sending a result
                    self.result_queue.put(request)
                    logger.info(f"Received result for task {request.get('task_id')} from {worker_id}")
                
                elif request.get("type") == "error":
                    # Worker encountered an error
                    logger.error(f"Error from worker {worker_id}: {request.get('error')}")
                    self.result_queue.put(request)
        
        except Exception as e:
            logger.error(f"Error handling worker {worker_id}: {str(e)}")
        
        finally:
            # Clean up
            if worker_id and worker_id in self.workers:
                del self.workers[worker_id]
            
            try:
                client_socket.close()
            except:
                pass
            
            logger.info(f"Worker {worker_id} disconnected")
    
    def add_task(self, task):
        """
        Add a task to the queue.
        
        Args:
            task: Task dictionary
        """
        self.task_queue.put(task)
        logger.info(f"Added task {task.get('id')} to queue")
    
    def get_result(self, timeout=None):
        """
        Get a result from the result queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class DistributedSystem:
    """
    Manages a distributed system of servers and workers.
    """
    def __init__(self, server_port=29501, is_server=False):
        """
        Initialize the distributed system.
        
        Args:
            server_port: Port for the task server
            is_server: Whether this instance is a server
        """
        self.server_port = server_port
        self.is_server = is_server
        self.server = None
        self.worker = None
    
    def start_server(self, max_workers=16):
        """
        Start a task server.
        
        Args:
            max_workers: Maximum number of workers to accept
        """
        if not self.is_server:
            logger.error("Cannot start server on a worker instance")
            return
        
        self.server = TaskServer(port=self.server_port, max_workers=max_workers)
        self.server.start()
    
    def start_worker(self, server_addr):
        """
        Start a worker that connects to the server.
        
        Args:
            server_addr: Address of the server
            
        Returns:
            True if worker started successfully, False otherwise
        """
        if self.is_server:
            logger.warning("Starting worker on server instance")
        
        self.worker = RemoteWorker(server_addr, self.server_port)
        if self.worker.connect():
            # Start worker in a thread
            worker_thread = threading.Thread(target=self.worker.run)
            worker_thread.daemon = True
            worker_thread.start()
            return True
        else:
            self.worker = None
            return False
    
    def stop(self):
        """
        Stop the system (server and/or worker).
        """
        if self.server:
            self.server.stop()
            self.server = None
        
        if self.worker:
            self.worker.disconnect()
            self.worker = None


def distributed_main(rank, world_size, args):
    """
    Main function for each distributed process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command-line arguments
    """
    # Import necessary modules
    from model import AlphaZeroChessNet
    from train import Trainer
    
    # Initialize distributed environment
    setup_worker(rank, world_size, args.master_addr, args.master_port)
    
    # Create model and move to GPU
    model = AlphaZeroChessNet()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" 
                          if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        device=device
    )
    
    # Train the model
    trainer.train(
        epochs=args.epochs,
        dataset_path=args.dataset_path,
        validate=True
    )
    
    # Clean up
    cleanup_worker()


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed AlphaZero")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run a task server")
    server_parser.add_argument("--port", type=int, default=29501, help="Server port")
    server_parser.add_argument("--max-workers", type=int, default=16, help="Maximum number of workers")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Run a worker")
    worker_parser.add_argument("--server", type=str, required=True, help="Server address")
    worker_parser.add_argument("--port", type=int, default=29501, help="Server port")
    
    # Distributed training command
    train_parser = subparsers.add_parser("train", help="Run distributed training")
    train_parser.add_argument("--world-size", type=int, default=1, help="Number of processes")
    train_parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    train_parser.add_argument("--master-port", type=int, default=29500, help="Master port")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset")
    train_parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    if args.command == "server":
        # Run a task server
        system = DistributedSystem(server_port=args.port, is_server=True)
        system.start_server(max_workers=args.max_workers)
        
        try:
            # Keep server running until Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
        finally:
            system.stop()
    
    elif args.command == "worker":
        # Run a worker
        system = DistributedSystem(server_port=args.port, is_server=False)
        if system.start_worker(args.server):
            try:
                # Keep worker running until Ctrl+C
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Worker stopping...")
            finally:
                system.stop()
    
    elif args.command == "train":
        # Run distributed training
        mp.spawn(
            distributed_main,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    
    else:
        parser.print_help() 