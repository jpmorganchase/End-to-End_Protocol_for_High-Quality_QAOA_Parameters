import math
import numpy as np
import torch 

def modify_elements(x, i, j, modify_i, modify_j):
    # Create a copy of the original tensor
    y = x.clone()

    # Get the elements at positions i and j
    y_i, y_j = x[i], x[j]

    # Modify the elements at positions i and j using non-in-place modification functions
    y[i] = modify_i(y_i, y_j)
    y[j] = modify_j(y_i, y_j)

    return y


def furxy(x: torch.Tensor, theta: torch.Tensor, q1: int, q2: int) -> torch.Tensor:
    """
    Applies e^{-i theta (XX + YY)} on q1, q2
    Same as XXPlusYYGate in Qiskit
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.XXPlusYYGate.html
    """
    
    if q1 > q2:
        q1, q2 = q2, q1
    
    n_states = len(x)
    n_groups = n_states // 4
    
    mask1 = (1<<q1)-1
    mask2 = (1<<(q2-1))-1
    maskm = mask1^mask2
    mask2 ^= (n_states-1)>>2
    
    wa = torch.cos(theta)
    wb = -1j*torch.sin(theta)
    for i in range(n_groups):
        i0 = (i&mask1) | ((i&maskm)<<1) | ((i&mask2)<<2)
        ia = i0 | (1<<q1)
        ib = i0 | (1<<q2)
        
        modify_ia = lambda x_ia, x_ib: wa * x_ia + wb * x_ib 
        modify_ib = lambda x_ia, x_ib: wb * x_ia + wa * x_ib 
        x = modify_elements(x, ia, ib, modify_ia, modify_ib)
        # x[ia], x[ib] = wa*x[ia]+wb*x[ib], wb*x[ia]+wa*x[ib]
        
    return x


def furxy_ring_torch(x: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:
    for i in range(2):
        for j in range(i, n_qubits-1, 2):
            x = furxy(x, theta, j, j+1)
    x = furxy(x, theta, 0, n_qubits-1)
    return x


def furxy_complete_torch(x: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:
    for i in range(n_qubits-1):
        for j in range(i+1, n_qubits):
            x = furxy(x, theta, i, j)

    return x
