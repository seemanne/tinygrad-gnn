from tinygrad.tensor import Tensor
from .utils import average

class MessagePassing:
    """Implements a basic message passing layer. 
    """
    def __init__(self, x, adjacency, aggr, message_func = None, propagation_func = None) -> None:
        self.x = x
        self.adjacency = adjacency
        self.aggr = aggr
        if not message_func: message_func = self._identity
        if not propagation_func: propagation_func = self._default_propagation

    def _default_propagation(self, x: Tensor, msg: Tensor):
        """Internal function that implements default aggregation which takes the mean of x and the messages.
        The Tensor gets reshaped internally.

        Args:
            x (Tensor): (len, ) shaped
            msg (Tensor): (msg, ) shaped

        Returns:
            Tensor: (len, ) shaped
        """
        x = x.reshape((1, x.shape[0]))
        msg = msg.reshape((1, msg.shape[0]))
        x_dash = x.cat(msg).mean(axis = 0)
        return x_dash


    def _identity(x: Tensor) -> Tensor:
        """Internal function that returns the input tensor. 
        Default message function if none is specified.

        Args:
            x (Tensor)

        Returns:
            Tensor: x
        """
        return x

    

    