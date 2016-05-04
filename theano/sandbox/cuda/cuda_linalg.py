from __future__ import absolute_import, print_function, division
import pkg_resources

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.fftconv import ScikitsCudaOp
from skcuda.rlinalg import rsvd

try:
    from theano.sandbox.cuda import cuda_ndarray, basic_ops, CudaNdarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cula_available = False

try:
    from scikits.cuda import cula
    cula_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

cula_initialized = False


class GpuRSVD(ScikitsCudaOp):
    """
    CULA GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('trans',)

    def __init__(self, method='standard', k=None, p=0,q=0):
        self.method = method
        self.k=k
        self.p=p
        self.q=q
        super(GpuRSVD, self).__init__()

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp):
        inp = basic_ops.gpu_contiguous(
            basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)(), CudaNdarrayType(broadcastable=[False] * 1)(), self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2):
        super(GpuRSVD, self).make_thunk(node, storage_map, _, _2)

        from theano.misc.pycuda_utils import to_gpuarray
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]


        def thunk():
            input_shape = inputs[0][0].shape   # m, n

            # construct output shape
            u_gpu_shape = (input_shape[0], self.k)
            s_gpu_shape = (self.k,)
            vt_gpu_shape = (self.k, input_shape[1])

            u_gpu = outputs[0]
            s_gpu =outputs[1]
            vt_gpu=outputs[2]

            # only allocate if there is no previous allocation of the
            # right size.
            for z, output_shape in ([u_gpu, u_gpu_shape], [s_gpu, s_gpu_shape], [vt_gpu, vt_gpu_shape]):
                if z[0] is None or z[0].shape != output_shape:
                    z[0] = CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])

            u_gpu_pycuda = to_gpuarray(u_gpu[0])
            s_gpu_pycuda = to_gpuarray(s_gpu[0])
            vt_gpu_pycuda = to_gpuarray(vt_gpu[0])

            u_gpu_pycuda[0], s_gpu_pycuda[0], vt_gpu_pycuda[0] = rsvd(input_pycuda, self.k, self.p, self.q, self.method)
        
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_rsvd = GpuRSVD()
