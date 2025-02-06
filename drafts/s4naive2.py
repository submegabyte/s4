import torch
import torch as torch.nn

def hippo(i, j):
    if i > j:
        return -(2*i+1)**0.5 * (2*j+1)**0.5
    return -(n+1) if n == k else 0

def hippo_matrix(n):
    ## https://chatgpt.com/share/67a326a3-a328-800c-ac33-3f9b786135d6
    ## https://chatgpt.com/share/67a3ed12-1698-800c-ad28-54b2506369a2
    rows = cols = n
    i, j = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    return hippo(i, j)

## 1-D input signal
## N-D latent state
## 1-D output signal
class S4Naive1D():
    def __init__(N=3, delta=1):
        self.A = nn.Parameter(hippo_matrix(N)) ## N x N
        self.B = nn.Parameter(torch.randn(N)) ## N x 1
        self.C = nn.Parameter(torch.randn(1, N)) ## 1 x N
        self.D = 0
        self.N = N
        self.delta = delta
        discretize(n)

    def discretize():
        I = torch.eye(N)
        Am = torch.inverse(I - self.delta/2 * self.A) ## N x N
        Ap = (I + delta/2 * self.A) ## N x N
        self.A1 = Am @ Ap ## N x N
        self.B1 = Am @ (self.delta * self.B) ## N x 1
        self.C1 = self.C ## 1 x N
    
    ## SSM convolution kernel/filter
    # def kernel():