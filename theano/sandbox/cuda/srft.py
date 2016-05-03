from __future__ import absolute_import, print_function, division
import string

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda import cuda_available, GpuOp
from theano.ifelse import ifelse
from theano.misc.pycuda_init import pycuda_available
from theano.sandbox.cuda.fftconv import ScikitsCudaOp

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray)
if pycuda_available:
    import pycuda.gpuarray

try:
    import scikits.cuda
    from scikits.cuda import fft, cublas
    scikits.cuda.misc.init()
    scikits_cuda_available = True
except (ImportError, Exception):
    scikits_cuda_available = False


# TODO: investigate the effect of enabling fastmath on FFT performance
# (how can it be enabled?).



class CuSRFT(ScikitsCudaOp):
    def output_type(self, inp):
        # add one extra dim for real/imag
        return CudaNdarrayType(
            broadcastable=[False] * (inp.type.ndim + 1))

    def make_thunk(self, node, storage_map, _, _2):
        super(CuSRFT, self).make_thunk(node, storage_map, _, _2)

        from theano.misc.pycuda_utils import to_gpuarray
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            output_shape = list(input_shape)
            # DFT of real input is symmetric, no need to store
            # redundant coefficients
            output_shape[-1] = output_shape[-1] // 2 + 1
            # extra dimension with length 2 for real/imag
            output_shape += [2]
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            # I thought we'd need to change the type on output_pycuda
            # so it is complex64, but as it turns out scikits.cuda.fft
            # doesn't really care either way and treats the array as
            # if it is complex64 anyway.
            output_pycuda = to_gpuarray(z[0])

            # only initialise plan if necessary
            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(input_shape[1:], np.float32, np.complex64,
                                   batch=input_shape[0])

            fft.fft(input_pycuda, output_pycuda, plan[0])

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


cusrft = CuSRFT()




