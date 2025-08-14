import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

# class ChebyshevLayer(nn.Module):
#     """Chebyshev polynomial operations for non-periodic boundaries"""
#     def __init__(self, in_channels: int, out_channels: int, modes: int):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes
        
#         # Learnable weights for Chebyshev coefficients
#         self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes))
#         nn.init.xavier_uniform_(self.weights)
    
#     def chebyshev_transform(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward Chebyshev transform (DCT-I)"""
#         return torch.fft.dct(x, type=1, dim=-1)
    
#     def chebyshev_inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
#         """Inverse Chebyshev transform (IDCT-I)"""
#         return torch.fft.idct(coeffs, type=1, dim=-1)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Transform to Chebyshev space
#         x_cheb = self.chebyshev_transform(x)
        
#         # Apply spectral convolution
#         out_cheb = torch.zeros_like(x_cheb[:, :self.out_channels])
#         out_cheb[..., :self.modes] = torch.einsum('bix,iox->box', 
#                                                  x_cheb[..., :self.modes], 
#                                                  self.weights)
        
#         # Transform back to physical space
#         return self.chebyshev_inverse(out_cheb)


from scipy.fftpack import dct, idct

class ChebyshevLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes))
        nn.init.xavier_uniform_(self.weights)

    def chebyshev_transform(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        x_dct = dct(x_np, type=1, axis=-1)
        return torch.from_numpy(x_dct).to(x.device)

    def chebyshev_inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_np = coeffs.detach().cpu().numpy()
        x_idct = idct(coeffs_np, type=1, axis=-1)
        return torch.from_numpy(x_idct).to(coeffs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cheb = self.chebyshev_transform(x)
        out_cheb = torch.zeros_like(x_cheb[:, :self.out_channels])
        out_cheb[..., :self.modes] = torch.einsum(
            'bix,iox->box', x_cheb[..., :self.modes], self.weights
        )
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
    
    def poisson_residual_spectral(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Compute Poisson equation residual: -∇²u - f"""
        if u.dim() == 3:  # 1D case: (batch, channels, x)
            laplacian = self.compute_derivatives_fourier(u, order=2, dim=-1)
        elif u.dim() == 4:  # 2D case: (batch, channels, x, y)
            d2u_dx2 = self.compute_derivatives_fourier(u, order=2, dim=-2)
            d2u_dy2 = self.compute_derivatives_fourier(u, order=2, dim=-1)
            laplacian = d2u_dx2 + d2u_dy2
        else:
            raise ValueError("Unsupported dimension")
        
        residual = -laplacian - f
        return residual
    
    def forward(self, prediction: torch.Tensor, source_term: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss"""
        residual = self.poisson_residual_spectral(prediction, source_term)
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
        self.loss_history = []
    
    def train_step(self, source_term: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        prediction = self.model(source_term)
        
        # Compute spectral loss
        loss = self.spectral_loss(prediction, source_term)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, source_terms: torch.Tensor, n_epochs: int = 1000):
        """Full training loop"""
        self.model.train()
        
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Training data shape: {source_terms.shape}")
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Simple batch training
            batch_size = min(8, source_terms.size(0))
            for i in range(0, source_terms.size(0), batch_size):
                batch = source_terms[i:i+batch_size].to(self.device)
                loss = self.train_step(batch)
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d}, Loss: {avg_loss:.8f}')
            
            # Step scheduler every 50 epochs
            if epoch % 50 == 0 and epoch > 0:
                self.scheduler.step()
        
        print(f"Training completed! Final loss: {self.loss_history[-1]:.8f}")
        return self.loss_history

class PDESolutionGenerator:
    """Generate analytical solutions for comparison"""
    
    @staticmethod
    def poisson_2d_analytical(source_func, domain_size: Tuple[float, float] = (2*np.pi, 2*np.pi),
                             grid_size: Tuple[int, int] = (64, 64)):
        """Generate analytical solution for 2D Poisson equation with periodic BC"""
        Lx, Ly = domain_size
        Nx, Ny = grid_size
        
        # Create coordinate grids
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Generate source term
        if callable(source_func):
            f = source_func(X, Y)
        else:
            f = source_func
        
        # Solve using spectral method
        f_hat = np.fft.fft2(f)
        
        # Create wavenumber grids
        kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx/Nx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, Ly/Ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Laplacian operator in Fourier space
        laplacian = -(KX**2 + KY**2)
        laplacian[0, 0] = 1  # Avoid division by zero
        
        # Solve for solution in Fourier space
        u_hat = f_hat / laplacian
        u_hat[0, 0] = 0  # Set mean to zero
        
        # Transform back to physical space
        u = np.real(np.fft.ifft2(u_hat))
        
        return X, Y, u, f

class NSMVisualizer:
    """Visualization tools for Neural Spectral Methods"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5)):
        self.figsize = figsize
        
    def plot_2d_solution(self, X, Y, u_nsm, u_true=None, title="NSM Solution"):
        """Plot 2D solution field(s) with optional error analysis"""
        
        if u_true is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        
        # NSM Solution
        im1 = axes[0].contourf(X, Y, u_nsm, levels=50, cmap='viridis')
        axes[0].set_title(f'{title} - NSM Solution')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        if u_true is not None:
            # True Solution
            im2 = axes[1].contourf(X, Y, u_true, levels=50, cmap='viridis')
            axes[1].set_title(f'{title} - True Solution')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1])
            
            # Error
            error = np.abs(u_nsm - u_true)
            im3 = axes[2].contourf(X, Y, error, levels=50, cmap='Reds')
            axes[2].set_title(f'{title} - Absolute Error')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[2])
            
            # Print error statistics
            l2_error = np.sqrt(np.mean(error**2))
            max_error = np.max(error)
            rel_error = l2_error / np.sqrt(np.mean(u_true**2))
            
            print(f"L2 Error: {l2_error:.6f}")
            print(f"Max Error: {max_error:.6f}")
            print(f"Relative Error: {rel_error:.6f}")
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_training_curve(self, loss_history, title="Training Loss"):
        """Plot training loss curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(loss_history, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_3d_surface(self, X, Y, u, title="3D Surface Plot"):
        """Create 3D surface plot"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, u, cmap='viridis', alpha=0.9,
                              linewidth=0, antialiased=True)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        
        fig.colorbar(surf, shrink=0.5, aspect=20)
        ax.view_init(elev=30, azim=45)
        
        plt.show()
        return fig

def generate_training_data(n_samples: int = 100, grid_size: int = 64):
    """Generate training data for 2D Poisson equation"""
    print(f"Generating {n_samples} training samples...")
    
    source_terms = []
    domain_size = (2*np.pi, 2*np.pi)
    
    for i in range(n_samples):
        # Random Fourier modes for source term
        def random_source(X, Y):
            result = np.zeros_like(X)
            n_modes = np.random.randint(2, 5)  # 2-4 modes
            for _ in range(n_modes):
                kx = np.random.randint(1, 4)  # wavenumber in x
                ky = np.random.randint(1, 4)  # wavenumber in y
                amp = np.random.uniform(0.5, 2.0)  # amplitude
                phase_x = np.random.uniform(0, 2*np.pi)
                phase_y = np.random.uniform(0, 2*np.pi)
                result += amp * np.sin(kx * X + phase_x) * np.cos(ky * Y + phase_y)
            return result
        
        # Generate source term
        x = np.linspace(0, domain_size[0], grid_size, endpoint=False)
        y = np.linspace(0, domain_size[1], grid_size, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = random_source(X, Y)
        
        source_terms.append(f)
    
    # Convert to tensor
    source_terms = np.array(source_terms)
    source_terms = torch.from_numpy(source_terms).float().unsqueeze(1)  # Add channel dimension
    
    print(f"Training data generated with shape: {source_terms.shape}")
    return source_terms

def main_training_pipeline():
    """Main training and visualization pipeline"""
    print("="*60)
    print("Neural Spectral Methods - Complete Training Pipeline")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    grid_size = 64
    n_samples = 50
    n_epochs = 500
    
    # Generate training data
    training_data = generate_training_data(n_samples, grid_size)
    
    # Create model
    print("\nCreating Neural Spectral Operator...")
    model = NeuralSpectralOperator(
        input_dim=1,      # Source term
        output_dim=1,     # Solution
        hidden_dim=64,
        n_layers=4,
        modes=[16, 16],   # Fourier modes
        # grid_type='fourier',
        grid_type='chebyshev',
        activation='gelu'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function and trainer
    spectral_loss = SpectralLoss(pde_type='poisson', grid_type='fourier')
    trainer = NSMTrainer(model, spectral_loss, learning_rate=1e-3, device=device)
    
    # Train the model
    print(f"\nTraining model for {n_epochs} epochs...")
    loss_history = trainer.train(training_data, n_epochs)
    
    # Create visualizer
    visualizer = NSMVisualizer()
    
    # Plot training curve
    print("\nPlotting training curve...")
    visualizer.plot_training_curve(loss_history, "NSM Training Loss")
    
    # Generate test case and compare solutions
    print("\nGenerating test case for comparison...")
    
    # Test source function
    def test_source(X, Y):
        return 2 * np.sin(2*X) * np.cos(Y) + np.sin(X + Y)
    
    # Generate analytical solution
    solution_generator = PDESolutionGenerator()
    X, Y, u_true, f_test = solution_generator.poisson_2d_analytical(
        test_source, grid_size=(grid_size, grid_size)
    )
    
    # Get NSM prediction
    model.eval()
    with torch.no_grad():
        f_test_tensor = torch.from_numpy(f_test).float().unsqueeze(0).unsqueeze(0).to(device)
        u_nsm_tensor = model(f_test_tensor)
        u_nsm = u_nsm_tensor.cpu().numpy().squeeze()
    
    # Visualize results
    print("\nGenerating comparison plots...")
    
    # 2D contour plots comparison
    visualizer.plot_2d_solution(X, Y, u_nsm, u_true, "2D Poisson Equation")
    
    # 3D surface plots
    visualizer.plot_3d_surface(X, Y, u_nsm, "NSM Solution - 3D View")
    visualizer.plot_3d_surface(X, Y, u_true, "True Solution - 3D View")
    
    # Additional test cases
    print("\nTesting on additional cases...")
    
    # Test case 2: Different source function
    def test_source2(X, Y):
        return np.sin(X) * np.sin(Y) + 0.5 * np.cos(3*X) * np.sin(2*Y)
    
    X2, Y2, u_true2, f_test2 = solution_generator.poisson_2d_analytical(
        test_source2, grid_size=(grid_size, grid_size)
    )
    
    with torch.no_grad():
        f_test2_tensor = torch.from_numpy(f_test2).float().unsqueeze(0).unsqueeze(0).to(device)
        u_nsm2_tensor = model(f_test2_tensor)
        u_nsm2 = u_nsm2_tensor.cpu().numpy().squeeze()
    
    visualizer.plot_2d_solution(X2, Y2, u_nsm2, u_true2, "2D Poisson - Test Case 2")
    
    print("\n" + "="*60)
    print("Training and visualization completed successfully!")
    print("="*60)
    
    return model, trainer, visualizer

if __name__ == "__main__":
    # Run the complete training and visualization pipeline
    main_training_pipeline()
