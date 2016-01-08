from __future__ import print_function

import logging



import theano
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType
from theano.tensor import basic as tensor
import numpy as np
from fjlt.random_projection_fast import fast_unitary_transform_fast_1d

logger = logging.getLogger(__name__)

class SRFT(Op):
    """
        Subsampled Randomized Fourrier Transform of a vector
    """
    __props__ = ('k', 'n')

    def __init__(self, k, n):
        self.k = k
        
        self.n = n
        self.D = np.sign(np.random.randn(self.n))
        self.srht_const = np.sqrt(self.n / self.k)
        self.S = np.random.choice(self.n, self.k, replace=False)


    def transform_1d(self, x):
        a = np.asarray(fast_unitary_transform_fast_1d(x.copy(), D=self.D))
        return self.srht_const * a[self.S]        

    def make_node(self, x):
        x = as_tensor_variable(x)

        assert x.ndim == 1, "x should be a vector."
        Px = theano.tensor.vector(dtype=x.dtype)

        return Apply(self, [x], [Px])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        Px = outputs[0]

        assert x.ndim == 1, "x should be a vector."
        Px[0] = self.transform_1d(x)
