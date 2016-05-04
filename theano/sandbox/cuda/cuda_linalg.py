from __future__ import absolute_import, print_function, division
import pkg_resources

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.fftconv import ScikitsCudaOp
from skcuda.rlinalg import rsvd

from pprint import pprint
from string import Template
from pycuda.tools import context_dependent_memoize
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from pycuda import curandom

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.elementwise as el
import pycuda.tools as tools
import numpy as np

from skcuda import cublas
from skcuda import misc
from skcuda import linalg

rand = curandom.MRG32k3aRandomNumberGenerator()

import sys
if sys.version_info < (3,):
    range = xrange



from skcuda.misc import init, add_matvec, div_matvec, mult_matvec
from skcuda.linalg import hermitian, transpose

# Get installation location of C headers:
from skcuda import install_headers

try:
    from theano.sandbox.cuda import cuda_ndarray, basic_ops, CudaNdarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cula_available = False

try:
    from skcuda import cula
    _has_cula = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    _has_cula = False

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
            s_gpu = outputs[1]
            vt_gpu = outputs[2]

            # only allocate if there is no previous allocation of the
            # right size.
            for z, output_shape in ([u_gpu, u_gpu_shape], [s_gpu, s_gpu_shape], [vt_gpu, vt_gpu_shape]):
                if z[0] is None or z[0].shape != output_shape:
                    #raise(NotImplementedError(str(output_shape)))
                    z[0] = CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])

            u_gpu_pycuda = to_gpuarray(u_gpu[0])
            s_gpu_pycuda = to_gpuarray(s_gpu[0])
            vt_gpu_pycuda = to_gpuarray(vt_gpu[0])
            
            rsvd_prealloc(input_pycuda,u_gpu_pycuda,s_gpu_pycuda,vt_gpu_pycuda, self.k, self.p, self.q, self.method)
        
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


def gpu_rsvd(a, method='standard', k=None, p=0,q=0):
    """
    This function performs the random SVD on GPU.

    Parameters
    ----------
    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 
    p : int
        `p` sets the oversampling parameter (default k=0).
    q : int
        `q` sets the number of power iterations (default=0).
    method : `{'standard', 'fast'}`
        'standard' : Standard algorithm as described in [1, 2]
        'fast' : Version II algorithm as described in [2]   


    Returns
    -------
    U, V,  D : matrices

    """
    return GpuRSVD(method, k, p, q)(a)


def rsvd_prealloc(a_gpu,U_gpu,s_gpu,Vt_gpu, k=None, p=0, q=0, method="standard", handle=None):
    """
    Randomized Singular Value Decomposition.
    
    Randomized algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `a` with target rank `k << n`. 
    The input matrix a is factored as `a = U * diag(s) * Vt`. The right singluar 
    vectors are the columns of the real or complex unitary matrix `U`. The left 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The paramter `p` is a oversampling parameter to improve the approximation. 
    A value between 2 and 10 is recommended.
    
    The paramter `q` specifies the number of normlized power iterations
    (subspace iterations) to reduce the approximation error. This is recommended 
    if the the singular values decay slowly and in practice 1 or 2 iterations 
    achive good results. However, computing power iterations is increasing the
    computational time. 
    
    If k > (n/1.5), partial SVD or trancated SVD might be faster. 
    
    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 
    p : int
        `p` sets the oversampling parameter (default k=0).
    q : int
        `q` sets the number of power iterations (default=0).
    method : `{'standard', 'fast'}`
        'standard' : Standard algorithm as described in [1, 2]
        'fast' : Version II algorithm as described in [2]   
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.                

    Returns
    -------
    u_gpu : pycuda.gpuarray
        Right singular values, array of shape `(m, k)`.
    s_gpu : pycuda.gpuarray
        Singular values, 1-d array of length `k`.
    vt_gpu : pycuda.gpuarray
        Left singular values, array of shape `(k, n)`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.
    
    Arrays are assumed to be stored in column-major order, i.e., order='F'.
    
    Input matrix of shape `(m, n)`, where `n>m` is not supported yet.

    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> from skcuda import linalg, rlinalg
    >>> linalg.init()
    >>> rlinalg.init()
    
    >>> #Randomized SVD decomposition of the square matrix `a` with single precision.
    >>> #Note: There is no gain to use rsvd if k > int(n/1.5)
    >>> a = np.array(np.random.randn(5, 5), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)  
    >>> U, s, Vt = rlinalg.rsvd(a_gpu, k=5, method='standard')
    >>> np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), 1e-4)
    True
    
    >>> #Low-rank SVD decomposition with target rank k=2
    >>> a = np.array(np.random.randn(5, 5), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)  
    >>> U, s, Vt = rlinalg.rsvd(a_gpu, k=2, method='standard')
    
    """
    
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                         <September, 2015>                         ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if data_type == np.complex64:
        cula_func_gesvd = cula.culaDeviceCgesvd
        cublas_func_gemm = cublas.cublasCgemm
        copy_func = cublas.cublasCcopy
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        TRANS_type = 'C'
        isreal = False
    elif data_type == np.float32:
        cula_func_gesvd = cula.culaDeviceSgesvd
        cublas_func_gemm = cublas.cublasSgemm
        copy_func = cublas.cublasScopy
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        TRANS_type = 'T'
        isreal = True
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func_gesvd = cula.culaDeviceZgesvd
                cublas_func_gemm = cublas.cublasZgemm
                copy_func = cublas.cublasZcopy
                alpha = np.complex128(1.0)
                beta = np.complex128(0.0)
                TRANS_type = 'C'
                isreal = False
            elif data_type == np.float64:
                cula_func_gesvd = cula.culaDeviceDgesvd
                cublas_func_gemm = cublas.cublasDgemm
                copy_func = cublas.cublasDcopy
                alpha = np.float64(1.0)
                beta = np.float64(0.0)
                TRANS_type = 'T'
                isreal = True
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    #CUDA assumes that arrays are stored in column-major order
    m, n = np.array(a_gpu.shape, int)
    if n>m : raise ValueError('input matrix of shape (m,n), where n>m is not supported')    
    
    #Set k 
    if k == None : raise ValueError('k must be provided')
    if k > n or k < 1: raise ValueError('k must be 0 < k <= n')
    kt = k
    k = k + p
    if k > n: k=n

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random sampling matrix O
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Allocate O 
    if isreal==False: 
        Oc_gpu = gpuarray.empty((n,k), real_type, order="F", allocator=alloc) 
    #End if
    O_gpu = gpuarray.empty((n,k), real_type, order="F", allocator=alloc) 

    #Draw random samples from a ~ Uniform(-1,1) distribution
    
    if isreal==True: 
        rand.fill_uniform(O_gpu)
        O_gpu = O_gpu * 2 - 1 #Scale to [-1,1]  
    else:
        rand.fill_uniform(O_gpu)
        rand.fill_uniform(Oc_gpu)

        O_gpu = (O_gpu + 1j * Oc_gpu) * 2 -1
    #End if
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * O
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #Allocate Y    
    Y_gpu = gpuarray.zeros((m,k), data_type, order="F", allocator=alloc)    
    #Dot product Y = A * O    
    cublas_func_gemm(handle, 'n', 'n', m, k, n, alpha, 
                         a_gpu.gpudata, m, O_gpu.gpudata, n, 
                         beta, Y_gpu.gpudata, m  )  
      
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #Note: economic QR just returns Q, and destroys Y_gpu
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

    if q > 0:
        Z_gpu = gpuarray.empty((n,k), data_type, order="F", allocator=alloc)    

        for i in np.arange(1, q+1 ):
            if( (2*i-2)%q == 0 ):
                Y_gpu = linalg.qr(Y_gpu, 'economic')
            
            cublas_func_gemm(handle, TRANS_type, 'n', n, k, m, alpha, 
                         a_gpu.gpudata, m, Y_gpu.gpudata, m, 
                         beta, Z_gpu.gpudata, n  )

            if( (2*i-1)%q == 0 ):
                Z_gpu = linalg.qr(Z_gpu, 'economic')
       
            cublas_func_gemm(handle, 'n', 'n', m, k, n, alpha, 
                         a_gpu.gpudata, m, Z_gpu.gpudata, n, 
                         beta, Y_gpu.gpudata, m  )
                         
        #End for
     #End if   
    
    Q_gpu = linalg.qr(Y_gpu, 'economic')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #Allocate B    
    B_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)    
    cublas_func_gemm(handle, TRANS_type, 'n', k, n, m, alpha, 
                         Q_gpu.gpudata, m, a_gpu.gpudata, m, 
                         beta, B_gpu.gpudata, k  )
    
    if method == 'standard':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition
        #Note: B = U" * S * Vt
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        #gesvd(jobu, jobvt, m, n, int(a), lda, int(s), int(u), ldu, int(vt), ldvt)
        #Allocate s, U, Vt for economic SVD
        #Note: singular values are always real
#        s_gpu = gpuarray.empty(k, real_type, order="F", allocator=alloc)
#         U_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
#         Vt_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)
        
        #Economic SVD
        cula_func_gesvd('S', 'S', k, n, int(B_gpu.gpudata), k, int(s_gpu.gpudata), 
                        int(U_gpu.gpudata), k, int(Vt_gpu.gpudata), k)
    
        #Compute right singular vectors as U = Q * U"
        cublas_func_gemm(handle, 'n', 'n', m, k, k, alpha, 
                         Q_gpu.gpudata, m, U_gpu.gpudata, k, 
                         beta, Q_gpu.gpudata, m  )
        U_gpu[0] =  Q_gpu   #Set pointer            

        # Free internal CULA memory:
        cula.culaFreeBuffers()      
         
        #Return
        #return U_gpu[ : , 0:kt ], s_gpu[ 0:kt ], Vt_gpu[ 0:kt , : ]
    
        
    elif method == 'fast':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Orthogonalize B.T using reduced QR decomposition: B.T = Q" * R"
        #Note: reduced QR returns Q and R, and destroys B_gpu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                
        if isreal==True: 
            B_gpu = transpose(B_gpu) #transpose B
        else:
            B_gpu = hermitian(B_gpu) #transpose B
        
        Qstar_gpu, Rstar_gpu = linalg.qr(B_gpu, 'reduced')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition of R"
        #Note: R" = U" * S" * Vt"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        #gesvd(jobu, jobvt, m, n, int(a), lda, int(s), int(u), ldu, int(vt), ldvt)
        #Allocate s, U, Vt for economic SVD
        #Note: singular values are always real
#        s_gpu = gpuarray.empty(k, real_type, order="F", allocator=alloc)
        Ustar_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
        Vtstar_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
        
        #Economic SVD
        cula_func_gesvd('A', 'A', k, k, int(Rstar_gpu.gpudata), k, int(s_gpu.gpudata), 
                        int(Ustar_gpu.gpudata), k, int(Vtstar_gpu.gpudata), k)
    
   
        #Compute right singular vectors as U = Q * Vt.T"
        cublas_func_gemm(handle, 'n', TRANS_type, m, k, k, alpha, 
                         Q_gpu.gpudata, m, Vtstar_gpu.gpudata, k, 
                         beta, Q_gpu.gpudata, m  )
        U_gpu[0] =  Q_gpu   #Set pointer  

        #Compute left singular vectors as Vt = U".T * Q".T  
        #Vt_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)

        cublas_func_gemm(handle, TRANS_type, TRANS_type, k, n, k, alpha, 
                         Ustar_gpu.gpudata, k, Qstar_gpu.gpudata, n, 
                         beta, Vt_gpu.gpudata, k  )
    
       

        # Free internal CULA memory:
        cula.culaFreeBuffers()      
         
        #Return
        #return U_gpu[ : , 0:kt ], s_gpu[ 0:kt ], Vt_gpu[ 0:kt , : ]    
    #End if
