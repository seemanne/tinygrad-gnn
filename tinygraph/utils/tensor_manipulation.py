from tinygrad.tensor import Tensor
import numpy as np
import functools as ft

def stack_columns(X: list[Tensor]) -> Tensor:
    """Stacks a list of 0d Tensors column wise.
    TODO test that confirms this is identity grad

    Args:
        X (list[Tensor]): List containing tensors to stack

    Returns:
        Tensor: stacked Tensor
    """
    X[0] = X[0].reshape((1,X[0].shape[0]))
    return ft.reduce(lambda x,y: x.cat(y.reshape((1, y.shape[0]))), X).transpose()

def stack_rows(X: list[Tensor]) -> Tensor:
    """Stacks a list of 0d Tensors row wise.

    Args:
        X (list[Tensor]): List containing tensors to stack

    Returns:
        Tensor: stacked Tensor
    """
    X[0] = X[0].reshape((1, X[0].shape[0]))
    return ft.reduce(lambda x,y: x.cat(y.reshape((1, y.shape[0]))), X)

