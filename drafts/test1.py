import math
import torch

def hippo(i, j):
    if i > j:
        return -(2*i+1)**0.5 * (2*j+1)**0.5
    return -(n+1) if n == k else 0

def hippo_matrix(n):
    ## https://chatgpt.com/share/67a326a3-a328-800c-ac33-3f9b786135d6
    rows = cols = n
    i, j = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    return hippo(i, j)

class S4():
    def __init__(n=3):
        self.A = hippo(n)
        self.B = torch.randn(n, n)
        self.C = torch.randn(n, n)
        discretize(n)

    def discretize(n=3):
        I = torch.eye(n)
        self.A1 = torch.inverse(I - self.delta/2 * self.A) @ (I + delta/2 * self.A)
        self.B1 = torch.inverse(I - self.delta/2 * self.A) @ self.delta @ self.B
        self.C1 = self.C
    
    ## SSM convolution kernel/filter
    # def kernel():