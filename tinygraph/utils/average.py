from tinygrad.tensor import Tensor
import numpy as np

def average(A: Tensor) -> Tensor:
    len = A.shape[1]
    x = Tensor(np.ones((len, 1)))
    y = A.matmul(x)
    return y

A = Tensor([[2, 1],[1, 0]], requires_grad= True)
y = average(A)

a = y.reshape(y.shape[0]).backward()
f = 'clown'