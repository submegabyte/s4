import torch
import torch.fft

## https://chatgpt.com/share/67a3bc02-3900-800c-a371-a74a38f446b5
class S4Convolution(torch.nn.Module):
    def __init__(self, N, L, step_size=0.1):
        """
        N: State size
        L: Sequence length
        step_size: Discretization step size
        """
        super().__init__()

        # S4 parameters
        self.N = N
        self.L = L
        self.step_size = step_size

        # Learnable parameters
        self.Lambda = torch.nn.Parameter(torch.randn(N, dtype=torch.cfloat))  # Diagonal part
        self.P = torch.nn.Parameter(torch.randn(N, 1, dtype=torch.cfloat))    # Low-rank factor
        self.Q = torch.nn.Parameter(torch.randn(N, 1, dtype=torch.cfloat))    # Low-rank factor
        self.B = torch.nn.Parameter(torch.randn(N, dtype=torch.cfloat))       # Input mapping
        self.C = torch.nn.Parameter(torch.randn(N, dtype=torch.cfloat))       # Output mapping

    def compute_kernel(self):
        """Compute the SSM convolution kernel using diagonalization and FFT"""
        # Compute the discrete-time version of A = Lambda - P Q^*
        A_disc = self.Lambda - self.P @ self.Q.conj().T
        
        # Compute the generating function in the frequency domain
        C_tilde = (torch.eye(self.N, self.L) - A_disc[:,:self.L]).conj().T @ self.C  # Truncated generating function

        # Compute the Cauchy kernel inversion using Woodbury identity
        # K_hat = (C_tilde / (1 - omega[None, :] @ self.Lambda[:, None])).sum(dim=0)

        ## Black box Cauchy kernel
        omega = torch.exp(2j * torch.pi * torch.arange(self.L) / self.L)  # FFT roots of unity
        k_omega = (C_tilde @ self.Q).conj().T @ torch.inverse(2/step_size * (1-omega)/(1+omega) - self.Lambda)

        # Convert back to time domain using Inverse FFT
        K = torch.fft.ifft(K_hat).real
        return K

    def forward(self, u):
        """
        Forward pass: Perform convolution of input u with kernel K.
        u: Input sequence of shape (batch, length)
        """
        K = self.compute_kernel()  # Compute convolution kernel
        return torch.nn.functional.conv1d(u.unsqueeze(1), K.unsqueeze(0).unsqueeze(0), padding=self.L-1).squeeze(1)

# Example usage
N, L = 64, 1024  # State size and sequence length
s4_layer = S4Convolution(N, L)
input_seq = torch.randn(1, L)  # Example input batch of size 1
output_seq = s4_layer(input_seq)
print(output_seq.shape)  # Should output (1, L)
