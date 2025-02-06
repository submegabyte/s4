## https://arxiv.org/pdf/2111.00396
## section C.3

## Convolution representation

import torch 
import torch.nn as nn

def ssmConv(Ad, Bd, Cd, L):
    Kd = torch.zeros(L)

    current_term = Bd ## N x P
    for i in range(L):
        Kd[i] = Cd.conj().T @ current_term ## 1 x 1
        current_term = Ad @ current_term ## N x 1
    
    return Kd

def ssmGen(z, Ad, Bd, Cd, L=None):
    ## (1 - x^n) = (1 - x) (1 + x + x^2 + ... + x^(n-1))
    
    N = Ad.shape[0]
    I = torch.eye(N)

    denominator = (I - Ad * z)

    if L is None:
        Kc = Cd.conj().T @ torch.inv(denominator) @ Bd
    else: ## truncated at length L
        numerator = (I - Ad**L * z**L)
        Kc = Cd.conj().T @ numerator @ torch.inv(denominator) @ Bd
    
    return Kc

## 1-D input signal
## N-D latent state
## 1-D output signal
class S4Conv1D(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        Lambda = nn.Parameter(torch.randn(N, N)) ## N x N
        r = 1
        P = nn.Parameter(torch.randn(N, r))
        Q = nn.Parameter(torch.randn(N, r))

        self.B = nn.Parameter(torch.randn(N, 1)) ## N x 1

        ## C is now its transposed conjugate
        ## B, C, P, Q have the same shape (N, 1)
        # self.C = nn.Parameter(torch.randn(1, N)) ## 1 x N
        self.C = nn.Parameter(torch.randn(N, 1)) ## N x 1

        self.D = 0 ## skip connection
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## scalar, step size

        ## discretize
        I = torch.eye(N)
        Ir = torch.eye(self.F)

        ## forward discretization
        A_0 = 2 / self.delta * I + (Lambda - P @ Q.conj().T)

        ## backward discretization
        D = torch.inverse(2 / self.delta - Lambda) ## N x N
        A_1 = D - D @ P @ torch.inverse(Ir + Q.conj().T @ D @ P) @ Q.conj().T @ D

        self.Ad = A_1 @ A_0 ## N x N
        self.Bd = 2 * A_1 @ self.B ## N x 1
        self.Cd = self.C ## N x 1 
    
    ## x: (L, P)
    def forward(self, x):
        h = torch.zeros(self.N)

        L = x.shape[0]

        for i in range(L):
            h = self.Ad @ h + self.Bd @ x[i]
        
        y = self.Cd.conj().T @ h
        return y

if __name__ == "__main__":
    L = 10
    F = 1 ## embedding
    N = 8 ## state

    x = torch.randn(L, F)
    model = S4Conv1D(N, F)

    y = model(x)

    print(y.shape) ## (1)