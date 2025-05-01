from optim.pso import ParticleSwarmOptimizer
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict

def train_pso(
    model,
    train_loader,
    criterion,
    variant="PPSO",
    num_particles=50,
    num_iterations=500,
    plot_convergence=True,
    print_every=25,
    val_loader=None,
    early_stopping_patience=None,
    run_id=None,
    config=None
):
    """
    Enhanced PSO training function with validation, early stopping, and better tracking
    
    Args:
        model: The model to optimize
        train_loader: Training data loader
        criterion: Loss function
        variant: PSO variant ('PPSO')
        num_particles: Number of particles in swarm
        num_iterations: Maximum iterations
        plot_convergence: Whether to plot learning curves
        print_every: Print frequency
        val_loader: Optional validation loader
        early_stopping_patience: Patience for early stopping
        run_id: Identifier for hyperparameter study
        config: Dictionary of PPSO parameters to override defaults
    """
    device = next(model.parameters()).device
    metrics = defaultdict(list)
    start_time = time.time()
    
    # Initialize PSO with optional config overrides
    optimizer = ParticleSwarmOptimizer(
        model=model,
        variant=variant,
        config=config,
        num_particles=num_particles,
        max_iter=num_iterations,
        device=device
    )

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_iterations):
        # Training step
        train_loss = optimizer.step(train_loader, criterion)
        if train_loss is None:  # Early stopping triggered
            break

        # Training metrics
        train_accuracy = evaluate_accuracy(model, train_loader, device)
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        
        # Validation metrics
        if val_loader is not None:
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
            metrics['val_loss'].append(val_loss)
            metrics['val_accuracy'].append(val_accuracy)
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                # Save best model parameters
                best_params = [p.detach().clone() for p in model.parameters()]
            else:
                epochs_no_improve += 1
                
            if (early_stopping_patience is not None and 
                epochs_no_improve >= early_stopping_patience):
                print(f"Early stopping at epoch {epoch}")
                # Restore best parameters
                for param, best_param in zip(model.parameters(), best_params):
                    param.data.copy_(best_param)
                break

        # Progress reporting
        if epoch % print_every == 0 or epoch == num_iterations - 1:
            log_str = (f"[{variant}] Run {run_id} | Epoch {epoch:03d}: "
                      f"Train Loss = {train_loss:.4f}, Acc = {train_accuracy:.4f}")
            if val_loader:
                log_str += (f" | Val Loss = {val_loss:.4f}, "
                           f"Val Acc = {val_accuracy:.4f} (Best: {best_val_accuracy:.4f})")
            print(log_str)

    # Final evaluation
    training_time = time.time() - start_time
    metrics['training_time'] = training_time
    
    if plot_convergence:
        plot_metrics(metrics, variant, run_id)

    # Return comprehensive results
    results = {
        'optimizer': optimizer,
        'best_position': optimizer.global_best_position,
        'metrics': dict(metrics),
        'config': {
            'variant': variant,
            'num_particles': num_particles,
            'cognitive_coeff': optimizer.params["cognitive_coeff"],
            'social_coeff': optimizer.params["social_coeff"],
            'inertia_range': optimizer.params["inertia_range"],
            'iterations': num_iterations,
            'run_id': run_id
        }
    }
    
    return results

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            total_loss += criterion(outputs, y_batch).item() * y_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
    return total_loss / total, correct / total

def evaluate_accuracy(model, data_loader, device):
    """Quick evaluation of accuracy only"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
    return correct / total

def plot_metrics(metrics, variant, run_id=None):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    if 'val_loss' in metrics:
        plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{variant} Loss\n(Run {run_id})' if run_id else f'{variant} Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracy'], label='Train Accuracy')
    if 'val_accuracy' in metrics:
        plt.plot(metrics['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{variant} Accuracy\n(Run {run_id})' if run_id else f'{variant} Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()