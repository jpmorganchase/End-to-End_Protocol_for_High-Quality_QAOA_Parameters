import math
import numpy as np


def furxy(x: np.ndarray, theta: float, q1: int, q2: int) -> np.ndarray:
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
    
    wa = math.cos(theta)
    wb = -1j*math.sin(theta)
    
    for i in range(n_groups):
        i0 = (i&mask1) | ((i&maskm)<<1) | ((i&mask2)<<2)
        ia = i0 | (1<<q1)
        ib = i0 | (1<<q2)
        x[ia], x[ib] = wa*x[ia]+wb*x[ib], wb*x[ia]+wa*x[ib]
        
    return x


def furxy_ring(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    for i in range(2):
        for j in range(i, n_qubits-1, 2):
            furxy(x, theta, j, j+1)
    furxy(x, theta, 0, n_qubits-1)

    return x


def furxy_complete(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    for i in range(n_qubits-1):
        for j in range(i+1, n_qubits):
            furxy(x, theta, i, j)

    return x
