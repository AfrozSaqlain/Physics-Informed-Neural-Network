import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('Outputs', exist_ok=True)

sys.stdout = open("Outputs/log.txt", "w", buffering=1)
sys.stderr = open("Outputs/error.err", "w", buffering=1)


class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), SineActivation(),
            nn.Linear(64, 64), SineActivation(),
            nn.Linear(64, 64), SineActivation(),
            nn.Linear(64, 1)
        )

    def forward(self, z, xi):
        inputs = torch.cat((z, xi), dim=1)
        return self.net(inputs)

# Define physics loss
def physics_loss(model, z, xi):
    z.requires_grad_(True)
    
    x = model(z, xi)
    dx_dz = torch.autograd.grad(x, z, torch.ones_like(x), create_graph=True)[0]
    d2x_dz2 = torch.autograd.grad(dx_dz, z, torch.ones_like(dx_dz), create_graph=True)[0]
    
    # Weighted residual to balance high and low damping solutions
    eq_residual = d2x_dz2 + 2 * xi * dx_dz + x
    weight = 1 / (1 + xi)  # Small xi gets larger weight
    return torch.mean(weight * eq_residual**2)

# Define initial condition loss for all ξ values
def initial_condition_loss(model, lambda_ic, xi_samples):
    z0 = torch.zeros((len(xi_samples), 1), dtype=torch.float32, requires_grad=True)  # z = 0 for all
    xi0 = xi_samples  # Use all sampled damping values

    x0_pred = model(z0, xi0)
    v0_pred = torch.autograd.grad(x0_pred, z0, torch.ones_like(x0_pred), create_graph=True)[0]

    ic_loss = torch.mean((x0_pred - 0.7) ** 2 + (v0_pred - 1.2) ** 2)
    return lambda_ic * ic_loss

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)

epochs = 20001
lambda_ic = 10

z_train = torch.linspace(0, 20, 200, dtype=torch.float32).view(-1, 1).to(device)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    xi_train = (torch.rand(200, 1) * 0.3 + 0.1).to(device)
    
    loss_pde = physics_loss(model, z_train, xi_train)
    loss_ic = initial_condition_loss(model, lambda_ic, xi_train)
    loss = loss_pde + loss_ic

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0 and lambda_ic > 1.0:
        lambda_ic *= 0.9

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}", flush=True)

# Define the function for the differential equations
def damped_oscillator(z, y, xi):
    x1, x2 = y  # y = [x, dx/dz]
    dx1_dz = x2
    dx2_dz = -2 * xi * x2 - x1
    return np.array([dx1_dz, dx2_dz])

# Runge-Kutta 4th order method for solving ODEs
def rk4_step(f, z, y, h, xi):
    k1 = h * f(z, y, xi)
    k2 = h * f(z + h/2, y + k1/2, xi)
    k3 = h * f(z + h/2, y + k2/2, xi)
    k4 = h * f(z + h, y + k3, xi)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Solve ODE numerically using RK4
def solve_damped_oscillator_rk4(x0, v0, xi, z_max, h):
    z_values = np.arange(0, z_max + h, h)
    y_values = np.zeros((len(z_values), 2))
    y_values[0] = [x0, v0]

    for i in range(1, len(z_values)):
        y_values[i] = rk4_step(damped_oscillator, z_values[i-1], y_values[i-1], h, xi)

    return z_values, y_values[:, 0]

x0, v0 = 0.7, 1.2
z_max, h = 20, 0.05


xi_test_vals = np.arange(0.1, 0.4, 0.05)  # Renamed to avoid overwriting in loop

num_plots = len(xi_test_vals)
cols = 2  # Set number of columns
rows = (num_plots + cols - 1) // cols  # Ensure enough rows

fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
axes = axes.flatten()  # Flatten in case of a multi-row grid

for i, xi_val in enumerate(xi_test_vals):  # Use a different loop variable

    z_values_rk4, x_values_rk4 = solve_damped_oscillator_rk4(x0, v0, xi_val, z_max, h)

    z_test = torch.linspace(0, 20, 100).view(-1, 1)
    xi_test = torch.full_like(z_test, xi_val)
    x_pred = model(z_test, xi_test).detach().numpy()

    ax = axes[i]
    ax.plot(z_values_rk4, x_values_rk4, 'r--', label='RK4 Numerical Solution')
    ax.plot(z_test.numpy(), x_pred, 'b-', label='PINN Solution')
    ax.set_xlabel('z')
    ax.set_ylabel('x(z)')
    ax.set_title(f'ξ = {xi_val:.2f}')
    ax.legend()
    ax.grid(True)

# Hide unused subplots if `num_plots` is not a perfect multiple of `cols`
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Damped_HO.png")
plt.show()
