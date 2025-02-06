## https://arxiv.org/pdf/2111.00396
## section C.1

import torch 
import torch.nn as nn

## p-D input signal
## N-D latent state
## p-D output signal
class S4Recurrent(nn.Module):
    def __init__(self, N=3, p=64, delta=1):
        super().__init__()

        ## Theorem 1
        ## NPLR (Normal Plus Low Rank)
        ## DPLR (Diagonal Plus Low Rank)
        ## A = Lambda - P Q*
        ## where
        ## A, Lambda: N x N
        ## P, Q: N x r
        ## r is the rank, can be 1 or 2
        Lambda = nn.Parameter(torch.randn(N, N)) ## N x N
        r = 1
        P = nn.Parameter(torch.randn(N, r))
        Q = nn.Parameter(torch.randn(N, r))

        ## must be set to a hippo matrix
        ## for our purposes, we omit that
        # self.A = nn.Parameter(torch.randn(N, N)) ## N x N


        self.B = nn.Parameter(torch.randn(N, p)) ## N x p
        self.C = nn.Parameter(torch.randn(p, N)) ## p x N
        self.D = 0 ## skip connection
        self.N = N ## state size
        self.p = p ## embedding length
        self.delta = delta ## scalar, step size

        ## discretize
        I = torch.eye(N)
        # Am = torch.inverse(I - self.delta/2 * self.A) ## N x N
        # Ap = (I + delta/2 * self.A) ## N x N

        ## forward discretization
        A_0 = 2 / self.delta * I + (Lambda - P @ Q.conj().T)

        ## backward discretization
        D = torch.inverse(2 / self.delta - Lambda) ## N x N
        Ir = torch.eye(r)
        A_1 = D - D @ P @ torch.inverse(Ir + Q.conj().T @ D @ P) @ Q.conj().T @ D

        self.A1 = A_1 @ A_0 ## N x N
        self.B1 = 2 * A_1 @ self.B ## N x P
        self.C1 = self.C ## P x N
    
    ## x: (L, P)
    def forward(self, x):
        h = torch.zeros(self.N)

        L = x.shape[0]

        for i in range(L):
            h = self.A1 @ h + self.B1 @ x[i]
        
        y = self.C1 @ h
        return y

if __name__ == "__main__":
    L = 10
    p = 64 ## embedding
    N = 8 ## state

    x = torch.randn(L, p)
    model = S4Recurrent(N, p)

    y = model(x)

    print(y.shape) ## (64)