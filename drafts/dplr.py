## https://chatgpt.com/share/67a445f5-d0b0-800c-b5a7-4b9c169a3b0b

import torch

def lambda_pq_decomposition(A, rank):
    """
    Decomposes a matrix A into the form A = Lambda - P Q*.

    Parameters:
        A (torch.Tensor): Input square matrix (n x n).
        rank (int): Rank of the low-rank correction.

    Returns:
        Lambda (torch.Tensor): Diagonal matrix (n x n).
        P (torch.Tensor): Low-rank left factor (n x rank).
        Q (torch.Tensor): Low-rank right factor (n x rank), already conjugate transposed.
    """
    n = A.shape[0]
    
    # Step 1: Extract the diagonal as Lambda
    Lambda = torch.diag(torch.diag(A))

    # Step 2: Compute the low-rank correction (A - Lambda)
    A_tilde = A - Lambda

    # Step 3: Compute SVD of A_tilde to obtain P and Q
    U, S, Vh = torch.linalg.svd(A_tilde, full_matrices=False)  # Vh is already conjugate transposed

    # Truncate to the specified rank
    P = U[:, :rank] * torch.sqrt(S[:rank])  # Scale with sqrt(S) for balance
    Q = (Vh[:rank, :].conj().T) * torch.sqrt(S[:rank])  # Conjugate transpose

    return Lambda, P, Q

# Example usage
torch.manual_seed(42)
n = 100  # Matrix size
rank = 10  # Low-rank correction rank

# Generate a random complex Hermitian matrix
A = torch.randn(n, n, dtype=torch.cfloat)
A = A + A.T.conj()  # Ensure Hermitian symmetry

# Compute the decomposition
Lambda, P, Q = lambda_pq_decomposition(A, rank)

# Verify the reconstruction
A_approx = Lambda - P @ Q.T.conj()
error = torch.norm(A - A_approx).item()

print("Original Matrix Shape:", A.shape)
print("Approximation Shape:", A_approx.shape)
print("Approximation Error:", error)