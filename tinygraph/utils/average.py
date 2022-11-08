from tinygrad.tensor import Tensor
import numpy as np

def average(A: Tensor) -> Tensor:
    """Averages the matrix column wise.
    Meant to be used as an aggregation function during aggregation steps.

    Args:
        A (Tensor): 2d Tensor with each row containing features from a different vertex.

    Returns:
        Tensor: Column-wise average of A
    """
    len = A.shape[0]
    x = Tensor(np.ones((len, 1)), requires_grad=True)
    y = A.transpose().matmul(x).div(len)
    return y



#A = Tensor([[2, 1],[1, 0], [0,1]], requires_grad= True)
#y = average(A)
