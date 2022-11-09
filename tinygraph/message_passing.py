from tinygrad.tensor import Tensor
from .utils import average
from .utils import tensor_manipulation

class MessagePassing:
    """Implements a basic message passing layer. 
    """
    def __init__(self, aggr = None, message_func = None, propagation_func = None) -> None:
        if not aggr: self.aggr = self._default_aggregation
        else: self.aggr = aggr

        if not message_func: self.message_func = self._identity
        else: self.message_func = message_func

        if not propagation_func: self.propagation_func = self._default_propagation
        else: self.propagation_func = propagation_func

    def _default_propagation(self, x: Tensor, msg: Tensor):
        """Internal function that implements default aggregation which takes the mean of x and the messages.
        The Tensor gets reshaped internally.

        Args:
            x (Tensor): (len, ) shaped
            msg (Tensor): (msg, ) shaped

        Returns:
            Tensor: (len, ) shaped
        """
        x_dash = tensor_manipulation.stack_rows([x, msg]).mean(axis = 0)
        return x_dash

    def _default_aggregation(self):
        return

    def _identity(x: Tensor) -> Tensor:
        """Internal function that returns the input tensor. 
        Default message function if none is specified.

        Args:
            x (Tensor)

        Returns:
            Tensor: x
        """
        return x

    

    