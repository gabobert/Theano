from __future__ import print_function

import logging



import theano
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType
from theano.tensor import basic as tensor 
import numpy as np
from fjlt.random_projection_fast import fast_unitary_transform_fast_1d32, fast_unitary_transform_fast_1d
_FLOATX = theano.config.floatX

logger = logging.getLogger(__name__)

class SRFT(Op):
    """
        Subsampled Randomized Fourrier Transform of a vector
    """
    __props__ = ('k', 'n')

    def __init__(self, k, n):#, wisdom_file='theano_wisdom'):
        self.k = k
        
        self.n = n
        self.D = np.sign(np.random.randn(self.n)).astype(_FLOATX)
        self.srht_const = np.sqrt(self.n / self.k).astype(_FLOATX)
        self.S = np.random.choice(self.n, self.k, replace=False)
        
#         self.wisdom_file = wisdom_file
#         try:
#             import_wisdom(self.wisdom_file)
#         except IOError:
#             print('wisdom file', self.wisdom_file, 'not found, starting new file.')

    def transform_1d64(self, x):
        a = np.asarray(fast_unitary_transform_fast_1d(x.copy(), D=self.D))
        return self.srht_const * a[self.S]     

    def transform_1d32(self, x):
        a = np.asarray(fast_unitary_transform_fast_1d32(x.copy(), D=self.D))
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
        if _FLOATX == 'float32':
            Px[0] = self.transform_1d32(x)
        elif _FLOATX == 'float64':
            Px[0] = self.transform_1d64(x)
        else:
            assert False, 'floatX needs to be float32 or float64'

#     def __dealloc__(self):
#         export_wisdom(self.wisdom_file)