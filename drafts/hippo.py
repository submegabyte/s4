## https://arxiv.org/pdf/2111.00396
## section C.1

import torch
import torch.nn as nn

def hippo_LagT(n, k):
    if n < k:
        return 0
    if n == k:
        return -0.5
    return -1

## rank = 1
def hippo_LegS(n, k):
    if n > k:
        return -(2*n+1)**0.5 * (2*k+1)**0.5
    if n == k:
        return -(n+1)
    return 0

def hippo_LegT(n, k):
    if n >= k:
        return -1
    return -(-1)**(n+k)
