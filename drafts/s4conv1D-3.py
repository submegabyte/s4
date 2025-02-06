## https://arxiv.org/pdf/2111.00396
## section C.3

## Convolution representation

import math
import torch
from torch import cfloat
import torch.nn as nn

def hippo(i, j):
    if i > j:
        return -(2*i+1)**0.5 * (2*j+1)**0.5
    return -(i+1) if i == j else 0

def hippo_matrix(N):
    A = torch.empty(N, N)

    for i in range(N):
        for j in range(N):
            A[i][j] = hippo(i, j)

    return A

def dplr(A):
    rank = 1
    Lambda = torch.diag(torch.diag(A))
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    P = U[:, :rank] * torch.sqrt(S[:rank])  # Scale with sqrt(S) for balance
    Q = (Vh[:rank, :].conj().T) * torch.sqrt(S[:rank])  # Conjugate transpose

    return Lambda, P, Q

## SSM Generating function with woodbury correction
## directly use delta, A, Ad, B, C instead of Ad, Bd, Cd
def ssmGen(delta, A, B, C, L):
    N = A.shape[0]
    I = torch.eye(N)
    I1 = torch.eye(1)

    ## C tilda star
    Ad = torch.inverse(I - delta/2 * A) @ (I + delta/2 * A)
    Cts = C.conj().T @ (I - Ad**L) ## 1 X N

    Lambda, P, Q = dplr(A)

    R = lambda z: torch.inverse(2/delta * (1-z)/(1+z) - Lambda) ## N x N
    QR = lambda z: Q.conj().T @ R(z) ## 1 x N

    ## woodbury identity
    ## 1 x 1 (wow, scalar?)
    Kcz = lambda z: 2/(1+z) * (Cts @ R(z) @ B - Cts @ R(z) @ P @ torch.inverse(I1 + QR(z) @ P) @ QR(z) @ B)

    ## ifft, Lemma C.2
    Kc = torch.empty(L, dtype=torch.cfloat)
    for i in range(L):
        Kc[i] = Kcz(torch.exp(torch.tensor(2 * math.pi * 1j * i / L)))
    
    return Kc

## SSM Convolution function using ifft
def ssmConv(delta, A, B, C, L):
    Kc = ssmGen(delta, A, B, C, L) ## (L)

    ## ifft
    Kd = torch.fft.ifft(Kc).real.to(torch.float32) ## (L)
    return Kd

## 1-D input signal
## N-D latent state
## 1-D output signal
class S4Conv1D(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        # Lambda = nn.Parameter(torch.randn(N, N)) ## N x N
        # # r = 1
        # P = nn.Parameter(torch.randn(N, r))
        # Q = nn.Parameter(torch.randn(N, r))

        self.A = nn.Parameter(hippo_matrix(N)).to(cfloat) ## N x N
        self.B = nn.Parameter(torch.randn(N, 1)).to(cfloat) ## N x 1

        ## C is now its transposed conjugate
        ## B, C, P, Q have the same shape (N, 1)
        # self.C = nn.Parameter(torch.randn(1, N)) ## 1 x N
        self.C = nn.Parameter(torch.randn(N, 1)).to(cfloat) ## N x 1

        self.D = 0 ## skip connection
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## scalar, step size
        
    ## x: (L)
    def forward(self, x):
        # h = torch.zeros(self.N)

        L = x.shape[0]

        Kd = ssmConv(self.delta, self.A, self.B, self.C, L)
        x = Kd @ x
        return x


if __name__ == "__main__":
    L = 10
    F = 1 ## embedding
    N = 8 ## state

    x = torch.randn(L, F)
    model = S4Conv1D(N, F)

    y = model(x)

    print(y.shape) ## (1)