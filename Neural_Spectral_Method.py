import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math

class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer"""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        # Initialize weights for Fourier modes
        # For rfft, we only need modes up to n//2 + 1
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes, 2))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization for complex weights
        nn.init.xavier_uniform_(self.weights[..., 0])
        nn.init.xavier_uniform_(self.weights[..., 1])
    
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 1D case"""
        # input: (batch, in_channel, x, 2), weights: (in_channel, out_channel, x, 2)
        return torch.stack([
            torch.einsum("bix,iox->box", input[..., 0], weights[..., 0]) - 
            torch.einsum("bix,iox->box", input[..., 1], weights[..., 1]),
            torch.einsum("bix,iox->box", input[..., 1], weights[..., 0]) + 
            torch.einsum("bix,iox->box", input[..., 0], weights[..., 1])
        ], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Convert to our complex representation
        x_ft_complex = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Multiply relevant Fourier modes
        out_ft_complex = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1, 2, 
                                    device=x.device, dtype=x.dtype)
        
        # Only use modes up to the minimum of self.modes and available modes
        max_modes = min(self.modes, x_ft_complex.size(-2))
        out_ft_complex[:, :, :max_modes] = self.compl_mul1d(
            x_ft_complex[:, :, :max_modes], self.weights[:, :, :max_modes]
        )
        
        # Convert back to torch complex
        out_ft = torch.complex(out_ft_complex[..., 0], out_ft_complex[..., 1])
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in first dimension
        self.modes2 = modes2  # Number of Fourier modes in second dimension
        
        # Initialize weights
        self.weights1 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights1[..., 0])
        nn.init.xavier_uniform_(self.weights1[..., 1])
        nn.init.xavier_uniform_(self.weights2[..., 0])
        nn.init.xavier_uniform_(self.weights2[..., 1])
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 2D case"""
        return torch.stack([
            torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 0]) - 
            torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 1]),
            torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 0]) + 
            torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 1])
        ], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        x_ft_complex = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Initialize output
        out_ft_complex = torch.zeros(batch_size, self.out_channels, 
                                    x.size(-2), x.size(-1)//2 + 1, 2,
                                    device=x.device, dtype=x.dtype)
        
        # Multiply relevant Fourier modes
        out_ft_complex[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft_complex[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft_complex[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft_complex[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Convert back to torch complex
        out_ft = torch.complex(out_ft_complex[..., 0], out_ft_complex[..., 1])
        
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[-2, -1])
        return x

class ChebyshevLayer(nn.Module):
    """Chebyshev polynomial operations for non-periodic boundaries"""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Learnable weights for Chebyshev coefficients
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes))
        nn.init.xavier_uniform_(self.weights)
    
    def chebyshev_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Chebyshev transform (DCT-I)"""
        return torch.fft.dct(x, type=1, dim=-1)
    
    def chebyshev_inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Inverse Chebyshev transform (IDCT-I)"""
        return torch.fft.idct(coeffs, type=1, dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform to Chebyshev space
        x_cheb = self.chebyshev_transform(x)
        
        # Apply spectral convolution
        out_cheb = torch.zeros_like(x_cheb[:, :self.out_channels])
        out_cheb[..., :self.modes] = torch.einsum('bix,iox->box', 
                                                 x_cheb[..., :self.modes], 
                                                 self.weights)
        
        # Transform back to physical space
        return self.chebyshev_inverse(out_cheb)

class NSMLayer(nn.Module):
    """Neural Spectral Method Layer"""
    def __init__(self, in_channels: int, out_channels: int, modes: List[int], 
                 grid_type: str = 'fourier', activation: str = 'gelu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.grid_type = grid_type
        
        # Spectral convolution layer
        if grid_type == 'fourier':
            if len(modes) == 1:
                self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes[0])
            elif len(modes) == 2:
                self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes[0], modes[1])
            else:
                raise NotImplementedError("Higher dimensional Fourier not implemented")
        elif grid_type == 'chebyshev':
            self.spectral_conv = ChebyshevLayer(in_channels, out_channels, modes[0])
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")
        
        # Linear layer for local operations
        self.linear = nn.Conv1d(in_channels, out_channels, 1) if len(modes) == 1 else \
                     nn.Conv2d(in_channels, out_channels, 1)
        
        # Activation function
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spectral convolution
        x1 = self.spectral_conv(x)
        
        # Apply local linear transformation
        x2 = self.linear(x)
        
        # Combine and activate
        out = x1 + x2
        out = self.activation(out)
        
        return out

class NeuralSpectralOperator(nn.Module):
    """Neural Spectral Method Operator"""
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 hidden_dim: int = 64,
                 n_layers: int = 4,
                 modes: List[int] = [32, 32],
                 grid_type: str = 'fourier',
                 activation: str = 'gelu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.modes = modes
        self.grid_type = grid_type
        
        # Input projection
        if len(modes) == 1:
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
            self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)
        else:
            self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
            self.output_proj = nn.Conv2d(hidden_dim, output_dim, 1)
        
        # NSM layers
        self.layers = nn.ModuleList([
            NSMLayer(hidden_dim, hidden_dim, modes, grid_type, activation)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Apply NSM layers
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class SpectralLoss(nn.Module):
    """Spectral Loss for PDE residuals"""
    def __init__(self, pde_type: str = 'poisson', grid_type: str = 'fourier'):
        super().__init__()
        self.pde_type = pde_type
        self.grid_type = grid_type
    
    def compute_derivatives_fourier(self, u: torch.Tensor, order: int = 1, dim: int = -1) -> torch.Tensor:
        """Compute derivatives using Fourier spectral differentiation"""
        # Get Fourier coefficients
        u_hat = torch.fft.fft(u, dim=dim)
        
        # Create wavenumbers
        n = u.size(dim)
        k = torch.fft.fftfreq(n, d=1.0/n, device=u.device)
        
        # Apply derivative operator in spectral space
        for _ in range(order):
            # Reshape k for broadcasting
            shape = [1] * u.dim()
            shape[dim] = -1
            k_reshaped = k.view(shape)
            
            u_hat = 1j * k_reshaped * u_hat
        
        # Transform back to physical space
        if order % 2 == 0:  # Even derivatives are real
            return torch.fft.ifft(u_hat, dim=dim).real
        else:  # Odd derivatives can be complex
            return torch.fft.ifft(u_hat, dim=dim).real
    
    def compute_derivatives_chebyshev(self, u: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute derivatives using Chebyshev spectral differentiation"""
        # Transform to Chebyshev space
        u_cheb = torch.fft.dct(u, type=1, dim=-1)
        
        # Apply differentiation in Chebyshev space
        n = u_cheb.size(-1)
        
        for _ in range(order):
            u_cheb_new = torch.zeros_like(u_cheb)
            for k in range(n-1):
                u_cheb_new[..., k] = 2 * (k+1) * u_cheb[..., k+1]
            u_cheb = u_cheb_new
        
        # Transform back to physical space
        return torch.fft.idct(u_cheb, type=1, dim=-1)
    
    def poisson_residual_spectral(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Compute Poisson equation residual: -∇²u - f"""
        if self.grid_type == 'fourier':
            if u.dim() == 3:  # 1D case: (batch, channels, x)
                laplacian = self.compute_derivatives_fourier(u, order=2, dim=-1)
            elif u.dim() == 4:  # 2D case: (batch, channels, x, y)
                d2u_dx2 = self.compute_derivatives_fourier(u, order=2, dim=-2)
                d2u_dy2 = self.compute_derivatives_fourier(u, order=2, dim=-1)
                laplacian = d2u_dx2 + d2u_dy2
            else:
                raise ValueError("Unsupported dimension")
        else:  # Chebyshev
            laplacian = self.compute_derivatives_chebyshev(u, order=2)
        
        residual = -laplacian - f
        return residual
    
    def reaction_diffusion_residual_spectral(self, u: torch.Tensor, u_init: torch.Tensor, 
                                           nu: float, rho: float = 5.0) -> torch.Tensor:
        """Compute reaction-diffusion residual: du/dt - ν*d²u/dx² - ρ*u*(1-u)"""
        if self.grid_type == 'fourier':
            du_dt = self.compute_derivatives_fourier(u, order=1, dim=-1)  # Time derivative
            d2u_dx2 = self.compute_derivatives_fourier(u, order=2, dim=-2)  # Spatial derivative
        else:
            du_dt = self.compute_derivatives_chebyshev(u, order=1)
            d2u_dx2 = self.compute_derivatives_chebyshev(u, order=2)
        
        # Nonlinear reaction term
        reaction = rho * u * (1 - u)
        
        residual = du_dt - nu * d2u_dx2 - reaction
        return residual
    
    def forward(self, prediction: torch.Tensor, *args) -> torch.Tensor:
        """Compute spectral loss"""
        if self.pde_type == 'poisson':
            source_term = args[0]
            residual = self.poisson_residual_spectral(prediction, source_term)
        elif self.pde_type == 'reaction_diffusion':
            u_init, nu = args
            residual = self.reaction_diffusion_residual_spectral(prediction, u_init, nu)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
        
        # Compute L2 norm of residual (Parseval's identity)
        if self.grid_type == 'fourier':
            # For Fourier basis, the spectral loss is just the L2 norm
            loss = torch.mean(residual ** 2)
        else:
            # For Chebyshev, we need to account for the weight function
            # This is a simplified version
            loss = torch.mean(residual ** 2)
        
        return loss

class NSMTrainer:
    """Training class for Neural Spectral Methods"""
    def __init__(self, model: NeuralSpectralOperator, 
                 spectral_loss: SpectralLoss,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.spectral_loss = spectral_loss
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
    
    def train_step(self, params: torch.Tensor, *args) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        prediction = self.model(params)
        
        # Compute spectral loss
        loss = self.spectral_loss(prediction, *args)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch: int = 0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            batch_data = [data.to(self.device) for data in batch_data]
            
            # Training step
            loss = self.train_step(*batch_data)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
        
        # Step scheduler
        self.scheduler.step()
        
        return avg_loss

# Example usage and demonstration
def create_poisson_example():
    """Create an example for 2D Poisson equation"""
    # Model parameters
    input_dim = 1  # Source term
    output_dim = 1  # Potential field
    hidden_dim = 64
    n_layers = 4
    modes = [32, 32]  # Fourier modes in x and y directions
    
    # Create model
    model = NeuralSpectralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        modes=modes,
        grid_type='fourier',
        activation='gelu'
    )
    
    # Create spectral loss
    loss_fn = SpectralLoss(pde_type='poisson', grid_type='fourier')
    
    # Create trainer
    trainer = NSMTrainer(model, loss_fn, learning_rate=1e-3)
    
    print("2D Poisson NSM model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, loss_fn, trainer

def create_reaction_diffusion_example():
    """Create an example for 1D Reaction-Diffusion equation"""
    # Model parameters  
    input_dim = 2  # Initial condition + parameter encoding
    output_dim = 1  # Solution u(x,t)
    hidden_dim = 64
    n_layers = 4
    modes = [64]  # Modes in spatial dimension
    
    # Create model
    model = NeuralSpectralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        modes=modes,
        grid_type='fourier',
        activation='gelu'
    )
    
    # Create spectral loss
    loss_fn = SpectralLoss(pde_type='reaction_diffusion', grid_type='fourier')
    
    # Create trainer
    trainer = NSMTrainer(model, loss_fn, learning_rate=1e-3)
    
    print("1D Reaction-Diffusion NSM model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, loss_fn, trainer

if __name__ == "__main__":
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Create and test Poisson model
    print("Creating 2D Poisson NSM example...")
    poisson_model, poisson_loss, poisson_trainer = create_poisson_example()
    
    # Test forward pass
    batch_size = 4
    grid_size = 64
    test_input = torch.randn(batch_size, 1, grid_size, grid_size).to(device)
    
    with torch.no_grad():
        test_output = poisson_model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {test_output.shape}")
        
        # Test loss computation
        test_loss = poisson_loss(test_output, test_input)
        print(f"Test loss: {test_loss.item():.6f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example: Create and test Reaction-Diffusion model
    print("Creating 1D Reaction-Diffusion NSM example...")
    rd_model, rd_loss, rd_trainer = create_reaction_diffusion_example()
    
    # Test forward pass
    test_input_rd = torch.randn(batch_size, 2, grid_size).to(device)
    
    with torch.no_grad():
        test_output_rd = rd_model(test_input_rd)
        print(f"Test input shape: {test_input_rd.shape}")
        print(f"Test output shape: {test_output_rd.shape}")
        
        # Test loss computation (dummy parameters)
        u_init = torch.randn_like(test_output_rd)
        nu = 0.01
        test_loss_rd = rd_loss(test_output_rd, u_init, nu)
        print(f"Test loss: {test_loss_rd.item():.6f}")
    
    print("\nNSM implementation completed successfully!")
