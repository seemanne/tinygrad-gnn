from tinygrad.tensor import Tensor
from utils import

class MessagePassing:
    """Implements a basic message passing layer. 
    """
    def __init__(self, x, adjacency, aggr, message_func = None, propagation_func = None) -> None:
        self.x = x
        self.adjacency = adjacency
        self.aggr = aggr
        if not message_func: message_func = self._identity
        if not propagation_func: propagation_func = self._default_propagation

    def _default_propagation(x: Tensor, msg: Tensor):
        


    def _identity(x: Tensor) -> Tensor:
        """Internal function that returns the input tensor. 
        Default message function if none is specified.

        Args:
            x (Tensor)

        Returns:
            Tensor: x
        """
        return x

    

    