from tinygrad.tensor import Tensor
from .utils import average
from .utils import tensor_manipulation

class MessagePassing:
    """Implements a basic message passing layer. 
    """
    def __init__(self, aggr_func = None, message_func = None, propagation_func = None) -> None:
        if not aggr_func: self.aggr_func = self._default_aggregation
        else: self.aggr_func = aggr_func

        if not message_func: self.message_func = self._identity
        else: self.message_func = message_func

        if not propagation_func: self.propagation_func = self._default_propagation
        else: self.propagation_func = propagation_func

    def forward(self, X: list[Tensor], adj_lists: list[list[Tensor]]):

        fX = [self.message_func(x) for x in X]
        Msg = [self.aggr_func(fX, adj_lists[i]) for i,_ in enumerate(fX)]
        X_dash = [self.propagation_func(X[i], Msg[i]) for i in range(len(X))]
        return X_dash

    def _default_propagation(self, x: Tensor, msg: Tensor):
        """Internal function that implements default propagation which takes the sum of x and the messages.

        Args:
            x (Tensor): (len, ) shaped
            msg (Tensor): (msg, ) shaped

        Returns:
            Tensor: (len, ) shaped
        """
        x_dash = tensor_manipulation.stack_rows([x, msg]).sum(axis = 0)
        return x_dash

    def _default_aggregation(self, fx_list: list[Tensor], adj_list: list[int]) -> Tensor:
        """Internal fuction that implements default aggregation which takes the sum of all messages

        Args:
            adj_list (list[int]): adj_list of receiving node
            fx_list (list[Tensor]): list containing features of all nodes

        Returns:
            Tensor: (len, ) shaped
        """

        msg = tensor_manipulation.stack_rows([fx_list[i] for i in adj_list])
        return msg.sum(axis = 0)

    def _identity(self, x: Tensor) -> Tensor:
        """Internal function that returns the input tensor. 
        Default message function if none is specified.

        Args:
            x (Tensor)

        Returns:
            Tensor: x
        """
        return x

    

    