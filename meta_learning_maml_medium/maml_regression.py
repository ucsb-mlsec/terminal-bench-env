# Complete MAML implementation
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def generate_sinusoid_task(amplitude=None, phase=None):
    """Generate a sinusoidal task with random or specified parameters."""
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5.0)
    if phase is None:
        phase = np.random.uniform(0, np.pi)
    
    def task_fn(x):
        return amplitude * np.sin(x + phase)
    
    return task_fn, amplitude, phase

def sample_task_data(task_fn, k_shot=10, query_size=10):
    """Sample support and query sets for a task."""
    x_support = np.random.uniform(-5, 5, (k_shot, 1))
    y_support = task_fn(x_support)
    
    x_query = np.random.uniform(-5, 5, (query_size, 1))
    y_query = task_fn(x_query)
    
    return (
        torch.FloatTensor(x_support),
        torch.FloatTensor(y_support),
        torch.FloatTensor(x_query),
        torch.FloatTensor(y_query)
    )

def maml_train(model, num_iterations=5000, inner_lr=0.01, meta_lr=0.001, num_tasks=5):
    """Meta-train the model using MAML."""
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    
    for iteration in range(num_iterations):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        for _ in range(num_tasks):
            # Sample a task
            task_fn, _, _ = generate_sinusoid_task()
            x_sup, y_sup, x_qry, y_qry = sample_task_data(task_fn, k_shot=10, query_size=10)
            
            # Inner loop: adapt to task
            adapted_params = {name: param.clone() for name, param in model.named_parameters()}
            
            for inner_step in range(5):
                # Forward pass with current adapted params
                y_pred = model(x_sup)
                inner_loss = nn.MSELoss()(y_pred, y_sup)
                
                # Compute gradients
                grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)
                
                # Update adapted params
                for (name, param), grad in zip(model.named_parameters(), grads):
                    adapted_params[name] = param - inner_lr * grad
            
            # Temporarily set adapted parameters
            original_params = {name: param.clone() for name, param in model.named_parameters()}
            for name, param in model.named_parameters():
                param.data = adapted_params[name].data
            
            # Compute meta loss on query set
            y_pred_query = model(x_qry)
            task_meta_loss = nn.MSELoss()(y_pred_query, y_qry)
            meta_loss += task_meta_loss
            
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data = original_params[name].data
        
        # Meta update
        meta_loss = meta_loss / num_tasks
        meta_loss.backward()
        meta_optimizer.step()
    
    return model

# Meta-train the model
model = RegressionNet()
model = maml_train(model)

# Evaluation on 5 test tasks
num_test_tasks = 5
before_losses = []
after_losses = []

for _ in range(num_test_tasks):
    # Generate a new test task
    task_fn, _, _ = generate_sinusoid_task()
    x_sup, y_sup, x_qry, y_qry = sample_task_data(task_fn, k_shot=10, query_size=10)
    
    # Measure initial MSE loss (before adaptation)
    with torch.no_grad():
        y_pred_before = model(x_qry)
        loss_before = nn.MSELoss()(y_pred_before, y_qry).item()
        before_losses.append(loss_before)
    
    # Adapt the model using 10 support examples with 3 gradient steps
    adapted_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Store original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    for step in range(3):
        # Forward pass
        y_pred = model(x_sup)
        loss = nn.MSELoss()(y_pred, y_sup)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        # Update parameters
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param.data = param.data - 0.01 * grad
    
    # Measure final MSE loss on query examples (after adaptation)
    with torch.no_grad():
        y_pred_after = model(x_qry)
        loss_after = nn.MSELoss()(y_pred_after, y_qry).item()
        after_losses.append(loss_after)
    
    # Restore original parameters for next task
    for name, param in model.named_parameters():
        param.data = original_params[name].data

# Compute average losses across all test tasks
avg_before = np.mean(before_losses)
avg_after = np.mean(after_losses)

# Save results to /workspace/results.txt
with open('/workspace/results.txt', 'w') as f:
    f.write(f'before={avg_before:.3f}\n')
    f.write(f'after={avg_after:.3f}\n')