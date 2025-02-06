import torch 
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

## P-D input signal
## N-D latent state
## P-D output signal
class S4Naive(nn.Module):
    def __init__(self, N=3, P=64, delta=1):
        super(S4Naive, self).__init__()
        self.A = nn.Parameter(hippo_matrix(N)) ## N x N
        self.B = nn.Parameter(torch.randn(N, P)) ## N x P
        self.C = nn.Parameter(torch.randn(P, N)) ## P x N
        self.D = 0 ## skip connection
        self.N = N ## state size
        self.P = P ## embedding length
        self.delta = delta ## scalar, step size

        ## discretize
        I = torch.eye(N)
        Am = torch.inverse(I - self.delta/2 * self.A) ## N x N
        Ap = (I + delta/2 * self.A) ## N x N
        self.A1 = Am @ Ap ## N x N
        self.B1 = Am @ (self.delta * self.B) ## N x P
        self.C1 = self.C ## P x N
    
    ## SSM convolution kernel/filter
    ## https://chatgpt.com/share/67a3e08a-4114-800c-b9fc-fa43ed8b01ea
    def kernel(self, L = 10):
        K1 = torch.zeros(L, P, P)

        current_term = self.B1 ## N x P
        for i in range(L):
            K1[i] = self.C1 @ current_term ## P x P
            current_term = self.A1 @ current_term ## N x P
        
        return K1 ## L, P, P
    
    ## x: (L, P)
    def forward(self, x):
        y = torch.zeros(self.P)

        L = x.shape[0]
        K1 = self.kernel(L)

        for i in range(L):
            y += K1[i] @ x[i]
        
        return y

if __name__ == "__main__":
    L = 10
    P = 64
    N = 8

    x = torch.randn(L, P)
    model = S4Naive(N, P)

    y = model(x)

    print(y.shape) ## (64)