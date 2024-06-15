from typing import List

class TensorRecurser:
  """
   To use this class, the child must define the following functions:
   1. self.getParts() to return the tensor-like objects to iterate over.
   2. self.createObject(results) to create a new instance of the object to be
      returned to the caller.
  """

  def __init__(self, *args, **kwargs):
    pass

  def new_tensor(self, *args, **kwargs) -> List:
    results = [ t.new_tensor(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def new_full(self, *args, **kwargs) -> List:
    results = [ t.new_full(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def new_empty(self, *args, **kwargs) -> List:
    results = [ t.new_empty(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def new_ones(self, *args, **kwargs) -> List:
    results = [ t.new_ones(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def new_zeros(self, *args, **kwargs) -> List:
    results = [ t.new_zeros(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_cuda(self, *args, **kwargs) -> List:
    results = [ t.is_cuda(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_quantized(self, *args, **kwargs) -> List:
    results = [ t.is_quantized(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_meta(self, *args, **kwargs) -> List:
    results = [ t.is_meta(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def device(self, *args, **kwargs) -> List:
    results = [ t.device(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def grad(self, *args, **kwargs) -> List:
    results = [ t.grad(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ndim(self, *args, **kwargs) -> List:
    results = [ t.ndim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def real(self, *args, **kwargs) -> List:
    results = [ t.real(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def imag(self, *args, **kwargs) -> List:
    results = [ t.imag(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nbytes(self, *args, **kwargs) -> List:
    results = [ t.nbytes(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def itemsize(self, *args, **kwargs) -> List:
    results = [ t.itemsize(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def abs(self, *args, **kwargs) -> List:
    results = [ t.abs(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def abs_(self, *args, **kwargs) -> List:
    results = [ t.abs_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def absolute(self, *args, **kwargs) -> List:
    results = [ t.absolute(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def absolute_(self, *args, **kwargs) -> List:
    results = [ t.absolute_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def acos(self, *args, **kwargs) -> List:
    results = [ t.acos(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def acos_(self, *args, **kwargs) -> List:
    results = [ t.acos_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arccos(self, *args, **kwargs) -> List:
    results = [ t.arccos(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arccos_(self, *args, **kwargs) -> List:
    results = [ t.arccos_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def add(self, *args, **kwargs) -> List:
    results = [ t.add(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def add_(self, *args, **kwargs) -> List:
    results = [ t.add_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addbmm(self, *args, **kwargs) -> List:
    results = [ t.addbmm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addbmm_(self, *args, **kwargs) -> List:
    results = [ t.addbmm_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addcdiv(self, *args, **kwargs) -> List:
    results = [ t.addcdiv(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addcdiv_(self, *args, **kwargs) -> List:
    results = [ t.addcdiv_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addcmul(self, *args, **kwargs) -> List:
    results = [ t.addcmul(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addcmul_(self, *args, **kwargs) -> List:
    results = [ t.addcmul_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addmm(self, *args, **kwargs) -> List:
    results = [ t.addmm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addmm_(self, *args, **kwargs) -> List:
    results = [ t.addmm_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sspaddmm(self, *args, **kwargs) -> List:
    results = [ t.sspaddmm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addmv(self, *args, **kwargs) -> List:
    results = [ t.addmv(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addmv_(self, *args, **kwargs) -> List:
    results = [ t.addmv_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addr(self, *args, **kwargs) -> List:
    results = [ t.addr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def addr_(self, *args, **kwargs) -> List:
    results = [ t.addr_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def adjoint(self, *args, **kwargs) -> List:
    results = [ t.adjoint(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def allclose(self, *args, **kwargs) -> List:
    results = [ t.allclose(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def amax(self, *args, **kwargs) -> List:
    results = [ t.amax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def amin(self, *args, **kwargs) -> List:
    results = [ t.amin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def aminmax(self, *args, **kwargs) -> List:
    results = [ t.aminmax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def angle(self, *args, **kwargs) -> List:
    results = [ t.angle(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def apply_(self, *args, **kwargs) -> List:
    results = [ t.apply_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def argmax(self, *args, **kwargs) -> List:
    results = [ t.argmax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def argmin(self, *args, **kwargs) -> List:
    results = [ t.argmin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def argsort(self, *args, **kwargs) -> List:
    results = [ t.argsort(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def argwhere(self, *args, **kwargs) -> List:
    results = [ t.argwhere(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def asin(self, *args, **kwargs) -> List:
    results = [ t.asin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def asin_(self, *args, **kwargs) -> List:
    results = [ t.asin_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arcsin(self, *args, **kwargs) -> List:
    results = [ t.arcsin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arcsin_(self, *args, **kwargs) -> List:
    results = [ t.arcsin_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def as_strided(self, *args, **kwargs) -> List:
    results = [ t.as_strided(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atan(self, *args, **kwargs) -> List:
    results = [ t.atan(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atan_(self, *args, **kwargs) -> List:
    results = [ t.atan_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctan(self, *args, **kwargs) -> List:
    results = [ t.arctan(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctan_(self, *args, **kwargs) -> List:
    results = [ t.arctan_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atan2(self, *args, **kwargs) -> List:
    results = [ t.atan2(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atan2_(self, *args, **kwargs) -> List:
    results = [ t.atan2_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctan2(self, *args, **kwargs) -> List:
    results = [ t.arctan2(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctan2_(self, *args, **kwargs) -> List:
    results = [ t.arctan2_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def all(self, *args, **kwargs) -> List:
    results = [ t.all(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def any(self, *args, **kwargs) -> List:
    results = [ t.any(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def backward(self, *args, **kwargs) -> List:
    results = [ t.backward(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def baddbmm(self, *args, **kwargs) -> List:
    results = [ t.baddbmm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def baddbmm_(self, *args, **kwargs) -> List:
    results = [ t.baddbmm_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bernoulli(self, *args, **kwargs) -> List:
    results = [ t.bernoulli(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bernoulli_(self, *args, **kwargs) -> List:
    results = [ t.bernoulli_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bfloat16(self, *args, **kwargs) -> List:
    results = [ t.bfloat16(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bincount(self, *args, **kwargs) -> List:
    results = [ t.bincount(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_not(self, *args, **kwargs) -> List:
    results = [ t.bitwise_not(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_not_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_not_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_and(self, *args, **kwargs) -> List:
    results = [ t.bitwise_and(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_and_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_and_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_or(self, *args, **kwargs) -> List:
    results = [ t.bitwise_or(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_or_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_or_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_xor(self, *args, **kwargs) -> List:
    results = [ t.bitwise_xor(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_xor_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_xor_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_left_shift(self, *args, **kwargs) -> List:
    results = [ t.bitwise_left_shift(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_left_shift_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_left_shift_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_right_shift(self, *args, **kwargs) -> List:
    results = [ t.bitwise_right_shift(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bitwise_right_shift_(self, *args, **kwargs) -> List:
    results = [ t.bitwise_right_shift_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bmm(self, *args, **kwargs) -> List:
    results = [ t.bmm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def bool(self, *args, **kwargs) -> List:
    results = [ t.bool(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def byte(self, *args, **kwargs) -> List:
    results = [ t.byte(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def broadcast_to(self, *args, **kwargs) -> List:
    results = [ t.broadcast_to(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cauchy_(self, *args, **kwargs) -> List:
    results = [ t.cauchy_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ceil(self, *args, **kwargs) -> List:
    results = [ t.ceil(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ceil_(self, *args, **kwargs) -> List:
    results = [ t.ceil_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def char(self, *args, **kwargs) -> List:
    results = [ t.char(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cholesky(self, *args, **kwargs) -> List:
    results = [ t.cholesky(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cholesky_inverse(self, *args, **kwargs) -> List:
    results = [ t.cholesky_inverse(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cholesky_solve(self, *args, **kwargs) -> List:
    results = [ t.cholesky_solve(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def chunk(self, *args, **kwargs) -> List:
    results = [ t.chunk(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def clamp(self, *args, **kwargs) -> List:
    results = [ t.clamp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def clamp_(self, *args, **kwargs) -> List:
    results = [ t.clamp_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def clip(self, *args, **kwargs) -> List:
    results = [ t.clip(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def clip_(self, *args, **kwargs) -> List:
    results = [ t.clip_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def clone(self, *args, **kwargs) -> List:
    results = [ t.clone(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def contiguous(self, *args, **kwargs) -> List:
    results = [ t.contiguous(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def copy_(self, *args, **kwargs) -> List:
    results = [ t.copy_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def conj(self, *args, **kwargs) -> List:
    results = [ t.conj(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def conj_physical(self, *args, **kwargs) -> List:
    results = [ t.conj_physical(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def conj_physical_(self, *args, **kwargs) -> List:
    results = [ t.conj_physical_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def resolve_conj(self, *args, **kwargs) -> List:
    results = [ t.resolve_conj(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def resolve_neg(self, *args, **kwargs) -> List:
    results = [ t.resolve_neg(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def copysign(self, *args, **kwargs) -> List:
    results = [ t.copysign(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def copysign_(self, *args, **kwargs) -> List:
    results = [ t.copysign_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cos(self, *args, **kwargs) -> List:
    results = [ t.cos(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cos_(self, *args, **kwargs) -> List:
    results = [ t.cos_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cosh(self, *args, **kwargs) -> List:
    results = [ t.cosh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cosh_(self, *args, **kwargs) -> List:
    results = [ t.cosh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def corrcoef(self, *args, **kwargs) -> List:
    results = [ t.corrcoef(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def count_nonzero(self, *args, **kwargs) -> List:
    results = [ t.count_nonzero(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cov(self, *args, **kwargs) -> List:
    results = [ t.cov(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def acosh(self, *args, **kwargs) -> List:
    results = [ t.acosh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def acosh_(self, *args, **kwargs) -> List:
    results = [ t.acosh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arccosh(self, *args, **kwargs) -> List:
    results = [ t.arccosh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arccosh_(self, *args, **kwargs) -> List:
    results = [ t.arccosh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cpu(self, *args, **kwargs) -> List:
    results = [ t.cpu(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cross(self, *args, **kwargs) -> List:
    results = [ t.cross(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cuda(self, *args, **kwargs) -> List:
    results = [ t.cuda(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logcumsumexp(self, *args, **kwargs) -> List:
    results = [ t.logcumsumexp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cummax(self, *args, **kwargs) -> List:
    results = [ t.cummax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cummin(self, *args, **kwargs) -> List:
    results = [ t.cummin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cumprod(self, *args, **kwargs) -> List:
    results = [ t.cumprod(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cumprod_(self, *args, **kwargs) -> List:
    results = [ t.cumprod_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cumsum(self, *args, **kwargs) -> List:
    results = [ t.cumsum(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cumsum_(self, *args, **kwargs) -> List:
    results = [ t.cumsum_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def chalf(self, *args, **kwargs) -> List:
    results = [ t.chalf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cfloat(self, *args, **kwargs) -> List:
    results = [ t.cfloat(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def cdouble(self, *args, **kwargs) -> List:
    results = [ t.cdouble(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def data_ptr(self, *args, **kwargs) -> List:
    results = [ t.data_ptr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def deg2rad(self, *args, **kwargs) -> List:
    results = [ t.deg2rad(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dequantize(self, *args, **kwargs) -> List:
    results = [ t.dequantize(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def det(self, *args, **kwargs) -> List:
    results = [ t.det(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dense_dim(self, *args, **kwargs) -> List:
    results = [ t.dense_dim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def detach(self, *args, **kwargs) -> List:
    results = [ t.detach(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def detach_(self, *args, **kwargs) -> List:
    results = [ t.detach_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diag(self, *args, **kwargs) -> List:
    results = [ t.diag(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diag_embed(self, *args, **kwargs) -> List:
    results = [ t.diag_embed(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diagflat(self, *args, **kwargs) -> List:
    results = [ t.diagflat(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diagonal(self, *args, **kwargs) -> List:
    results = [ t.diagonal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diagonal_scatter(self, *args, **kwargs) -> List:
    results = [ t.diagonal_scatter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fill_diagonal_(self, *args, **kwargs) -> List:
    results = [ t.fill_diagonal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fmax(self, *args, **kwargs) -> List:
    results = [ t.fmax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fmin(self, *args, **kwargs) -> List:
    results = [ t.fmin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def diff(self, *args, **kwargs) -> List:
    results = [ t.diff(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def digamma(self, *args, **kwargs) -> List:
    results = [ t.digamma(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def digamma_(self, *args, **kwargs) -> List:
    results = [ t.digamma_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dim(self, *args, **kwargs) -> List:
    results = [ t.dim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dim_order(self, *args, **kwargs) -> List:
    results = [ t.dim_order(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dist(self, *args, **kwargs) -> List:
    results = [ t.dist(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def div(self, *args, **kwargs) -> List:
    results = [ t.div(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def div_(self, *args, **kwargs) -> List:
    results = [ t.div_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def divide(self, *args, **kwargs) -> List:
    results = [ t.divide(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def divide_(self, *args, **kwargs) -> List:
    results = [ t.divide_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dot(self, *args, **kwargs) -> List:
    results = [ t.dot(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def double(self, *args, **kwargs) -> List:
    results = [ t.double(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def dsplit(self, *args, **kwargs) -> List:
    results = [ t.dsplit(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def element_size(self, *args, **kwargs) -> List:
    results = [ t.element_size(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def eq(self, *args, **kwargs) -> List:
    results = [ t.eq(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def eq_(self, *args, **kwargs) -> List:
    results = [ t.eq_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def equal(self, *args, **kwargs) -> List:
    results = [ t.equal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erf(self, *args, **kwargs) -> List:
    results = [ t.erf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erf_(self, *args, **kwargs) -> List:
    results = [ t.erf_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erfc(self, *args, **kwargs) -> List:
    results = [ t.erfc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erfc_(self, *args, **kwargs) -> List:
    results = [ t.erfc_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erfinv(self, *args, **kwargs) -> List:
    results = [ t.erfinv(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def erfinv_(self, *args, **kwargs) -> List:
    results = [ t.erfinv_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def exp(self, *args, **kwargs) -> List:
    results = [ t.exp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def exp_(self, *args, **kwargs) -> List:
    results = [ t.exp_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def expm1(self, *args, **kwargs) -> List:
    results = [ t.expm1(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def expm1_(self, *args, **kwargs) -> List:
    results = [ t.expm1_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def expand(self, *args, **kwargs) -> List:
    results = [ t.expand(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def expand_as(self, *args, **kwargs) -> List:
    results = [ t.expand_as(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def exponential_(self, *args, **kwargs) -> List:
    results = [ t.exponential_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fix(self, *args, **kwargs) -> List:
    results = [ t.fix(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fix_(self, *args, **kwargs) -> List:
    results = [ t.fix_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fill_(self, *args, **kwargs) -> List:
    results = [ t.fill_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def flatten(self, *args, **kwargs) -> List:
    results = [ t.flatten(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def flip(self, *args, **kwargs) -> List:
    results = [ t.flip(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fliplr(self, *args, **kwargs) -> List:
    results = [ t.fliplr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def flipud(self, *args, **kwargs) -> List:
    results = [ t.flipud(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def float(self, *args, **kwargs) -> List:
    results = [ t.float(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def float_power(self, *args, **kwargs) -> List:
    results = [ t.float_power(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def float_power_(self, *args, **kwargs) -> List:
    results = [ t.float_power_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def floor(self, *args, **kwargs) -> List:
    results = [ t.floor(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def floor_(self, *args, **kwargs) -> List:
    results = [ t.floor_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def floor_divide(self, *args, **kwargs) -> List:
    results = [ t.floor_divide(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def floor_divide_(self, *args, **kwargs) -> List:
    results = [ t.floor_divide_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fmod(self, *args, **kwargs) -> List:
    results = [ t.fmod(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def fmod_(self, *args, **kwargs) -> List:
    results = [ t.fmod_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def frac(self, *args, **kwargs) -> List:
    results = [ t.frac(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def frac_(self, *args, **kwargs) -> List:
    results = [ t.frac_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def frexp(self, *args, **kwargs) -> List:
    results = [ t.frexp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def gather(self, *args, **kwargs) -> List:
    results = [ t.gather(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def gcd(self, *args, **kwargs) -> List:
    results = [ t.gcd(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def gcd_(self, *args, **kwargs) -> List:
    results = [ t.gcd_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ge(self, *args, **kwargs) -> List:
    results = [ t.ge(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ge_(self, *args, **kwargs) -> List:
    results = [ t.ge_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def greater_equal(self, *args, **kwargs) -> List:
    results = [ t.greater_equal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def greater_equal_(self, *args, **kwargs) -> List:
    results = [ t.greater_equal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def geometric_(self, *args, **kwargs) -> List:
    results = [ t.geometric_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def geqrf(self, *args, **kwargs) -> List:
    results = [ t.geqrf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ger(self, *args, **kwargs) -> List:
    results = [ t.ger(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def get_device(self, *args, **kwargs) -> List:
    results = [ t.get_device(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def gt(self, *args, **kwargs) -> List:
    results = [ t.gt(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def gt_(self, *args, **kwargs) -> List:
    results = [ t.gt_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def greater(self, *args, **kwargs) -> List:
    results = [ t.greater(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def greater_(self, *args, **kwargs) -> List:
    results = [ t.greater_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def half(self, *args, **kwargs) -> List:
    results = [ t.half(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def hardshrink(self, *args, **kwargs) -> List:
    results = [ t.hardshrink(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def heaviside(self, *args, **kwargs) -> List:
    results = [ t.heaviside(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def histc(self, *args, **kwargs) -> List:
    results = [ t.histc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def histogram(self, *args, **kwargs) -> List:
    results = [ t.histogram(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def hsplit(self, *args, **kwargs) -> List:
    results = [ t.hsplit(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def hypot(self, *args, **kwargs) -> List:
    results = [ t.hypot(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def hypot_(self, *args, **kwargs) -> List:
    results = [ t.hypot_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def i0(self, *args, **kwargs) -> List:
    results = [ t.i0(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def i0_(self, *args, **kwargs) -> List:
    results = [ t.i0_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def igamma(self, *args, **kwargs) -> List:
    results = [ t.igamma(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def igamma_(self, *args, **kwargs) -> List:
    results = [ t.igamma_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def igammac(self, *args, **kwargs) -> List:
    results = [ t.igammac(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def igammac_(self, *args, **kwargs) -> List:
    results = [ t.igammac_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_add_(self, *args, **kwargs) -> List:
    results = [ t.index_add_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_add(self, *args, **kwargs) -> List:
    results = [ t.index_add(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_copy_(self, *args, **kwargs) -> List:
    results = [ t.index_copy_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_copy(self, *args, **kwargs) -> List:
    results = [ t.index_copy(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_fill_(self, *args, **kwargs) -> List:
    results = [ t.index_fill_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_fill(self, *args, **kwargs) -> List:
    results = [ t.index_fill(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_put_(self, *args, **kwargs) -> List:
    results = [ t.index_put_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_put(self, *args, **kwargs) -> List:
    results = [ t.index_put(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_reduce_(self, *args, **kwargs) -> List:
    results = [ t.index_reduce_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_reduce(self, *args, **kwargs) -> List:
    results = [ t.index_reduce(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def index_select(self, *args, **kwargs) -> List:
    results = [ t.index_select(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def indices(self, *args, **kwargs) -> List:
    results = [ t.indices(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def inner(self, *args, **kwargs) -> List:
    results = [ t.inner(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def int(self, *args, **kwargs) -> List:
    results = [ t.int(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def int_repr(self, *args, **kwargs) -> List:
    results = [ t.int_repr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def inverse(self, *args, **kwargs) -> List:
    results = [ t.inverse(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isclose(self, *args, **kwargs) -> List:
    results = [ t.isclose(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isfinite(self, *args, **kwargs) -> List:
    results = [ t.isfinite(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isinf(self, *args, **kwargs) -> List:
    results = [ t.isinf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isposinf(self, *args, **kwargs) -> List:
    results = [ t.isposinf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isneginf(self, *args, **kwargs) -> List:
    results = [ t.isneginf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isnan(self, *args, **kwargs) -> List:
    results = [ t.isnan(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_contiguous(self, *args, **kwargs) -> List:
    results = [ t.is_contiguous(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_complex(self, *args, **kwargs) -> List:
    results = [ t.is_complex(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_conj(self, *args, **kwargs) -> List:
    results = [ t.is_conj(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_floating_point(self, *args, **kwargs) -> List:
    results = [ t.is_floating_point(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_inference(self, *args, **kwargs) -> List:
    results = [ t.is_inference(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_leaf(self, *args, **kwargs) -> List:
    results = [ t.is_leaf(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_pinned(self, *args, **kwargs) -> List:
    results = [ t.is_pinned(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_set_to(self, *args, **kwargs) -> List:
    results = [ t.is_set_to(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_shared(self, *args, **kwargs) -> List:
    results = [ t.is_shared(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_signed(self, *args, **kwargs) -> List:
    results = [ t.is_signed(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def is_sparse(self, *args, **kwargs) -> List:
    results = [ t.is_sparse(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def istft(self, *args, **kwargs) -> List:
    results = [ t.istft(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def isreal(self, *args, **kwargs) -> List:
    results = [ t.isreal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def item(self, *args, **kwargs) -> List:
    results = [ t.item(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def kthvalue(self, *args, **kwargs) -> List:
    results = [ t.kthvalue(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lcm(self, *args, **kwargs) -> List:
    results = [ t.lcm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lcm_(self, *args, **kwargs) -> List:
    results = [ t.lcm_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ldexp(self, *args, **kwargs) -> List:
    results = [ t.ldexp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ldexp_(self, *args, **kwargs) -> List:
    results = [ t.ldexp_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def le(self, *args, **kwargs) -> List:
    results = [ t.le(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def le_(self, *args, **kwargs) -> List:
    results = [ t.le_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def less_equal(self, *args, **kwargs) -> List:
    results = [ t.less_equal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def less_equal_(self, *args, **kwargs) -> List:
    results = [ t.less_equal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lerp(self, *args, **kwargs) -> List:
    results = [ t.lerp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lerp_(self, *args, **kwargs) -> List:
    results = [ t.lerp_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lgamma(self, *args, **kwargs) -> List:
    results = [ t.lgamma(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lgamma_(self, *args, **kwargs) -> List:
    results = [ t.lgamma_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log(self, *args, **kwargs) -> List:
    results = [ t.log(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log_(self, *args, **kwargs) -> List:
    results = [ t.log_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logdet(self, *args, **kwargs) -> List:
    results = [ t.logdet(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log10(self, *args, **kwargs) -> List:
    results = [ t.log10(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log10_(self, *args, **kwargs) -> List:
    results = [ t.log10_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log1p(self, *args, **kwargs) -> List:
    results = [ t.log1p(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log1p_(self, *args, **kwargs) -> List:
    results = [ t.log1p_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log2(self, *args, **kwargs) -> List:
    results = [ t.log2(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log2_(self, *args, **kwargs) -> List:
    results = [ t.log2_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def log_normal_(self, *args, **kwargs) -> List:
    results = [ t.log_normal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logaddexp(self, *args, **kwargs) -> List:
    results = [ t.logaddexp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logaddexp2(self, *args, **kwargs) -> List:
    results = [ t.logaddexp2(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logsumexp(self, *args, **kwargs) -> List:
    results = [ t.logsumexp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_and(self, *args, **kwargs) -> List:
    results = [ t.logical_and(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_and_(self, *args, **kwargs) -> List:
    results = [ t.logical_and_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_not(self, *args, **kwargs) -> List:
    results = [ t.logical_not(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_not_(self, *args, **kwargs) -> List:
    results = [ t.logical_not_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_or(self, *args, **kwargs) -> List:
    results = [ t.logical_or(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_or_(self, *args, **kwargs) -> List:
    results = [ t.logical_or_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_xor(self, *args, **kwargs) -> List:
    results = [ t.logical_xor(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logical_xor_(self, *args, **kwargs) -> List:
    results = [ t.logical_xor_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logit(self, *args, **kwargs) -> List:
    results = [ t.logit(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def logit_(self, *args, **kwargs) -> List:
    results = [ t.logit_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def long(self, *args, **kwargs) -> List:
    results = [ t.long(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lt(self, *args, **kwargs) -> List:
    results = [ t.lt(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lt_(self, *args, **kwargs) -> List:
    results = [ t.lt_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def less(self, *args, **kwargs) -> List:
    results = [ t.less(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def less_(self, *args, **kwargs) -> List:
    results = [ t.less_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lu(self, *args, **kwargs) -> List:
    results = [ t.lu(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def lu_solve(self, *args, **kwargs) -> List:
    results = [ t.lu_solve(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def as_subclass(self, *args, **kwargs) -> List:
    results = [ t.as_subclass(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def map_(self, *args, **kwargs) -> List:
    results = [ t.map_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def masked_scatter_(self, *args, **kwargs) -> List:
    results = [ t.masked_scatter_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def masked_scatter(self, *args, **kwargs) -> List:
    results = [ t.masked_scatter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def masked_fill_(self, *args, **kwargs) -> List:
    results = [ t.masked_fill_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def masked_fill(self, *args, **kwargs) -> List:
    results = [ t.masked_fill(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def masked_select(self, *args, **kwargs) -> List:
    results = [ t.masked_select(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def matmul(self, *args, **kwargs) -> List:
    results = [ t.matmul(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def matrix_power(self, *args, **kwargs) -> List:
    results = [ t.matrix_power(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def matrix_exp(self, *args, **kwargs) -> List:
    results = [ t.matrix_exp(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def max(self, *args, **kwargs) -> List:
    results = [ t.max(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def maximum(self, *args, **kwargs) -> List:
    results = [ t.maximum(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mean(self, *args, **kwargs) -> List:
    results = [ t.mean(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def module_load(self, *args, **kwargs) -> List:
    results = [ t.module_load(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nanmean(self, *args, **kwargs) -> List:
    results = [ t.nanmean(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def median(self, *args, **kwargs) -> List:
    results = [ t.median(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nanmedian(self, *args, **kwargs) -> List:
    results = [ t.nanmedian(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def min(self, *args, **kwargs) -> List:
    results = [ t.min(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def minimum(self, *args, **kwargs) -> List:
    results = [ t.minimum(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mm(self, *args, **kwargs) -> List:
    results = [ t.mm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def smm(self, *args, **kwargs) -> List:
    results = [ t.smm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mode(self, *args, **kwargs) -> List:
    results = [ t.mode(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def movedim(self, *args, **kwargs) -> List:
    results = [ t.movedim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def moveaxis(self, *args, **kwargs) -> List:
    results = [ t.moveaxis(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def msort(self, *args, **kwargs) -> List:
    results = [ t.msort(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mul(self, *args, **kwargs) -> List:
    results = [ t.mul(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mul_(self, *args, **kwargs) -> List:
    results = [ t.mul_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def multiply(self, *args, **kwargs) -> List:
    results = [ t.multiply(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def multiply_(self, *args, **kwargs) -> List:
    results = [ t.multiply_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def multinomial(self, *args, **kwargs) -> List:
    results = [ t.multinomial(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mv(self, *args, **kwargs) -> List:
    results = [ t.mv(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mvlgamma(self, *args, **kwargs) -> List:
    results = [ t.mvlgamma(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def mvlgamma_(self, *args, **kwargs) -> List:
    results = [ t.mvlgamma_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nansum(self, *args, **kwargs) -> List:
    results = [ t.nansum(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def narrow(self, *args, **kwargs) -> List:
    results = [ t.narrow(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def narrow_copy(self, *args, **kwargs) -> List:
    results = [ t.narrow_copy(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ndimension(self, *args, **kwargs) -> List:
    results = [ t.ndimension(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nan_to_num(self, *args, **kwargs) -> List:
    results = [ t.nan_to_num(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nan_to_num_(self, *args, **kwargs) -> List:
    results = [ t.nan_to_num_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ne(self, *args, **kwargs) -> List:
    results = [ t.ne(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ne_(self, *args, **kwargs) -> List:
    results = [ t.ne_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def not_equal(self, *args, **kwargs) -> List:
    results = [ t.not_equal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def not_equal_(self, *args, **kwargs) -> List:
    results = [ t.not_equal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def neg(self, *args, **kwargs) -> List:
    results = [ t.neg(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def neg_(self, *args, **kwargs) -> List:
    results = [ t.neg_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def negative(self, *args, **kwargs) -> List:
    results = [ t.negative(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def negative_(self, *args, **kwargs) -> List:
    results = [ t.negative_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nelement(self, *args, **kwargs) -> List:
    results = [ t.nelement(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nextafter(self, *args, **kwargs) -> List:
    results = [ t.nextafter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nextafter_(self, *args, **kwargs) -> List:
    results = [ t.nextafter_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nonzero(self, *args, **kwargs) -> List:
    results = [ t.nonzero(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def norm(self, *args, **kwargs) -> List:
    results = [ t.norm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def normal_(self, *args, **kwargs) -> List:
    results = [ t.normal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def numel(self, *args, **kwargs) -> List:
    results = [ t.numel(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def numpy(self, *args, **kwargs) -> List:
    results = [ t.numpy(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def orgqr(self, *args, **kwargs) -> List:
    results = [ t.orgqr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ormqr(self, *args, **kwargs) -> List:
    results = [ t.ormqr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def outer(self, *args, **kwargs) -> List:
    results = [ t.outer(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def permute(self, *args, **kwargs) -> List:
    results = [ t.permute(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def pin_memory(self, *args, **kwargs) -> List:
    results = [ t.pin_memory(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def pinverse(self, *args, **kwargs) -> List:
    results = [ t.pinverse(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def polygamma(self, *args, **kwargs) -> List:
    results = [ t.polygamma(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def polygamma_(self, *args, **kwargs) -> List:
    results = [ t.polygamma_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def positive(self, *args, **kwargs) -> List:
    results = [ t.positive(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def pow(self, *args, **kwargs) -> List:
    results = [ t.pow(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def pow_(self, *args, **kwargs) -> List:
    results = [ t.pow_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def prod(self, *args, **kwargs) -> List:
    results = [ t.prod(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def put_(self, *args, **kwargs) -> List:
    results = [ t.put_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def qr(self, *args, **kwargs) -> List:
    results = [ t.qr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def qscheme(self, *args, **kwargs) -> List:
    results = [ t.qscheme(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def quantile(self, *args, **kwargs) -> List:
    results = [ t.quantile(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def nanquantile(self, *args, **kwargs) -> List:
    results = [ t.nanquantile(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def q_scale(self, *args, **kwargs) -> List:
    results = [ t.q_scale(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def q_zero_point(self, *args, **kwargs) -> List:
    results = [ t.q_zero_point(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def q_per_channel_scales(self, *args, **kwargs) -> List:
    results = [ t.q_per_channel_scales(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def q_per_channel_zero_points(self, *args, **kwargs) -> List:
    results = [ t.q_per_channel_zero_points(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def q_per_channel_axis(self, *args, **kwargs) -> List:
    results = [ t.q_per_channel_axis(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def rad2deg(self, *args, **kwargs) -> List:
    results = [ t.rad2deg(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def random_(self, *args, **kwargs) -> List:
    results = [ t.random_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def ravel(self, *args, **kwargs) -> List:
    results = [ t.ravel(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def reciprocal(self, *args, **kwargs) -> List:
    results = [ t.reciprocal(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def reciprocal_(self, *args, **kwargs) -> List:
    results = [ t.reciprocal_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def record_stream(self, *args, **kwargs) -> List:
    results = [ t.record_stream(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def register_hook(self, *args, **kwargs) -> List:
    results = [ t.register_hook(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def register_post_accumulate_grad_hook(self, *args, **kwargs) -> List:
    results = [ t.register_post_accumulate_grad_hook(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def remainder(self, *args, **kwargs) -> List:
    results = [ t.remainder(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def remainder_(self, *args, **kwargs) -> List:
    results = [ t.remainder_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def renorm(self, *args, **kwargs) -> List:
    results = [ t.renorm(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def renorm_(self, *args, **kwargs) -> List:
    results = [ t.renorm_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def repeat(self, *args, **kwargs) -> List:
    results = [ t.repeat(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def repeat_interleave(self, *args, **kwargs) -> List:
    results = [ t.repeat_interleave(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def requires_grad(self, *args, **kwargs) -> List:
    results = [ t.requires_grad(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def requires_grad_(self, *args, **kwargs) -> List:
    results = [ t.requires_grad_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def reshape(self, *args, **kwargs) -> List:
    results = [ t.reshape(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def reshape_as(self, *args, **kwargs) -> List:
    results = [ t.reshape_as(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def resize_(self, *args, **kwargs) -> List:
    results = [ t.resize_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def resize_as_(self, *args, **kwargs) -> List:
    results = [ t.resize_as_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def retain_grad(self, *args, **kwargs) -> List:
    results = [ t.retain_grad(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def retains_grad(self, *args, **kwargs) -> List:
    results = [ t.retains_grad(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def roll(self, *args, **kwargs) -> List:
    results = [ t.roll(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def rot90(self, *args, **kwargs) -> List:
    results = [ t.rot90(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def round(self, *args, **kwargs) -> List:
    results = [ t.round(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def round_(self, *args, **kwargs) -> List:
    results = [ t.round_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def rsqrt(self, *args, **kwargs) -> List:
    results = [ t.rsqrt(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def rsqrt_(self, *args, **kwargs) -> List:
    results = [ t.rsqrt_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter(self, *args, **kwargs) -> List:
    results = [ t.scatter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter_(self, *args, **kwargs) -> List:
    results = [ t.scatter_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter_add_(self, *args, **kwargs) -> List:
    results = [ t.scatter_add_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter_add(self, *args, **kwargs) -> List:
    results = [ t.scatter_add(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter_reduce_(self, *args, **kwargs) -> List:
    results = [ t.scatter_reduce_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def scatter_reduce(self, *args, **kwargs) -> List:
    results = [ t.scatter_reduce(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def select(self, *args, **kwargs) -> List:
    results = [ t.select(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def select_scatter(self, *args, **kwargs) -> List:
    results = [ t.select_scatter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def set_(self, *args, **kwargs) -> List:
    results = [ t.set_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def share_memory_(self, *args, **kwargs) -> List:
    results = [ t.share_memory_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def short(self, *args, **kwargs) -> List:
    results = [ t.short(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sigmoid(self, *args, **kwargs) -> List:
    results = [ t.sigmoid(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sigmoid_(self, *args, **kwargs) -> List:
    results = [ t.sigmoid_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sign(self, *args, **kwargs) -> List:
    results = [ t.sign(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sign_(self, *args, **kwargs) -> List:
    results = [ t.sign_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def signbit(self, *args, **kwargs) -> List:
    results = [ t.signbit(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sgn(self, *args, **kwargs) -> List:
    results = [ t.sgn(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sgn_(self, *args, **kwargs) -> List:
    results = [ t.sgn_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sin(self, *args, **kwargs) -> List:
    results = [ t.sin(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sin_(self, *args, **kwargs) -> List:
    results = [ t.sin_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sinc(self, *args, **kwargs) -> List:
    results = [ t.sinc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sinc_(self, *args, **kwargs) -> List:
    results = [ t.sinc_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sinh(self, *args, **kwargs) -> List:
    results = [ t.sinh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sinh_(self, *args, **kwargs) -> List:
    results = [ t.sinh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def asinh(self, *args, **kwargs) -> List:
    results = [ t.asinh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def asinh_(self, *args, **kwargs) -> List:
    results = [ t.asinh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arcsinh(self, *args, **kwargs) -> List:
    results = [ t.arcsinh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arcsinh_(self, *args, **kwargs) -> List:
    results = [ t.arcsinh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def shape(self, *args, **kwargs) -> List:
    results = [ t.shape(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def size(self, *args, **kwargs) -> List:
    results = [ t.size(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def slogdet(self, *args, **kwargs) -> List:
    results = [ t.slogdet(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def slice_scatter(self, *args, **kwargs) -> List:
    results = [ t.slice_scatter(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def softmax(self, *args, **kwargs) -> List:
    results = [ t.softmax(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sort(self, *args, **kwargs) -> List:
    results = [ t.sort(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def split(self, *args, **kwargs) -> List:
    results = [ t.split(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sparse_mask(self, *args, **kwargs) -> List:
    results = [ t.sparse_mask(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sparse_dim(self, *args, **kwargs) -> List:
    results = [ t.sparse_dim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sqrt(self, *args, **kwargs) -> List:
    results = [ t.sqrt(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sqrt_(self, *args, **kwargs) -> List:
    results = [ t.sqrt_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def square(self, *args, **kwargs) -> List:
    results = [ t.square(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def square_(self, *args, **kwargs) -> List:
    results = [ t.square_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def squeeze(self, *args, **kwargs) -> List:
    results = [ t.squeeze(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def squeeze_(self, *args, **kwargs) -> List:
    results = [ t.squeeze_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def std(self, *args, **kwargs) -> List:
    results = [ t.std(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def stft(self, *args, **kwargs) -> List:
    results = [ t.stft(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def storage(self, *args, **kwargs) -> List:
    results = [ t.storage(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def untyped_storage(self, *args, **kwargs) -> List:
    results = [ t.untyped_storage(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def storage_offset(self, *args, **kwargs) -> List:
    results = [ t.storage_offset(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def storage_type(self, *args, **kwargs) -> List:
    results = [ t.storage_type(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def stride(self, *args, **kwargs) -> List:
    results = [ t.stride(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sub(self, *args, **kwargs) -> List:
    results = [ t.sub(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sub_(self, *args, **kwargs) -> List:
    results = [ t.sub_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def subtract(self, *args, **kwargs) -> List:
    results = [ t.subtract(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def subtract_(self, *args, **kwargs) -> List:
    results = [ t.subtract_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sum(self, *args, **kwargs) -> List:
    results = [ t.sum(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def sum_to_size(self, *args, **kwargs) -> List:
    results = [ t.sum_to_size(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def svd(self, *args, **kwargs) -> List:
    results = [ t.svd(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def swapaxes(self, *args, **kwargs) -> List:
    results = [ t.swapaxes(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def swapdims(self, *args, **kwargs) -> List:
    results = [ t.swapdims(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def t(self, *args, **kwargs) -> List:
    results = [ t.t(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def t_(self, *args, **kwargs) -> List:
    results = [ t.t_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tensor_split(self, *args, **kwargs) -> List:
    results = [ t.tensor_split(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tile(self, *args, **kwargs) -> List:
    results = [ t.tile(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to(self, *args, **kwargs) -> List:
    results = [ t.to(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_mkldnn(self, *args, **kwargs) -> List:
    results = [ t.to_mkldnn(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def take(self, *args, **kwargs) -> List:
    results = [ t.take(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def take_along_dim(self, *args, **kwargs) -> List:
    results = [ t.take_along_dim(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tan(self, *args, **kwargs) -> List:
    results = [ t.tan(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tan_(self, *args, **kwargs) -> List:
    results = [ t.tan_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tanh(self, *args, **kwargs) -> List:
    results = [ t.tanh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tanh_(self, *args, **kwargs) -> List:
    results = [ t.tanh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atanh(self, *args, **kwargs) -> List:
    results = [ t.atanh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def atanh_(self, *args, **kwargs) -> List:
    results = [ t.atanh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctanh(self, *args, **kwargs) -> List:
    results = [ t.arctanh(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def arctanh_(self, *args, **kwargs) -> List:
    results = [ t.arctanh_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tolist(self, *args, **kwargs) -> List:
    results = [ t.tolist(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def topk(self, *args, **kwargs) -> List:
    results = [ t.topk(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_dense(self, *args, **kwargs) -> List:
    results = [ t.to_dense(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_sparse(self, *args, **kwargs) -> List:
    results = [ t.to_sparse(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_sparse_csr(self, *args, **kwargs) -> List:
    results = [ t.to_sparse_csr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_sparse_csc(self, *args, **kwargs) -> List:
    results = [ t.to_sparse_csc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_sparse_bsr(self, *args, **kwargs) -> List:
    results = [ t.to_sparse_bsr(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def to_sparse_bsc(self, *args, **kwargs) -> List:
    results = [ t.to_sparse_bsc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def trace(self, *args, **kwargs) -> List:
    results = [ t.trace(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def transpose(self, *args, **kwargs) -> List:
    results = [ t.transpose(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def transpose_(self, *args, **kwargs) -> List:
    results = [ t.transpose_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def triangular_solve(self, *args, **kwargs) -> List:
    results = [ t.triangular_solve(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tril(self, *args, **kwargs) -> List:
    results = [ t.tril(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def tril_(self, *args, **kwargs) -> List:
    results = [ t.tril_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def triu(self, *args, **kwargs) -> List:
    results = [ t.triu(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def triu_(self, *args, **kwargs) -> List:
    results = [ t.triu_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def true_divide(self, *args, **kwargs) -> List:
    results = [ t.true_divide(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def true_divide_(self, *args, **kwargs) -> List:
    results = [ t.true_divide_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def trunc(self, *args, **kwargs) -> List:
    results = [ t.trunc(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def trunc_(self, *args, **kwargs) -> List:
    results = [ t.trunc_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def type(self, *args, **kwargs) -> List:
    results = [ t.type(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def type_as(self, *args, **kwargs) -> List:
    results = [ t.type_as(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unbind(self, *args, **kwargs) -> List:
    results = [ t.unbind(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unflatten(self, *args, **kwargs) -> List:
    results = [ t.unflatten(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unfold(self, *args, **kwargs) -> List:
    results = [ t.unfold(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def uniform_(self, *args, **kwargs) -> List:
    results = [ t.uniform_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unique(self, *args, **kwargs) -> List:
    results = [ t.unique(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unique_consecutive(self, *args, **kwargs) -> List:
    results = [ t.unique_consecutive(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unsqueeze(self, *args, **kwargs) -> List:
    results = [ t.unsqueeze(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def unsqueeze_(self, *args, **kwargs) -> List:
    results = [ t.unsqueeze_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def values(self, *args, **kwargs) -> List:
    results = [ t.values(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def var(self, *args, **kwargs) -> List:
    results = [ t.var(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def vdot(self, *args, **kwargs) -> List:
    results = [ t.vdot(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def view(self, *args, **kwargs) -> List:
    results = [ t.view(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def view_as(self, *args, **kwargs) -> List:
    results = [ t.view_as(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def vsplit(self, *args, **kwargs) -> List:
    results = [ t.vsplit(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def where(self, *args, **kwargs) -> List:
    results = [ t.where(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def xlogy(self, *args, **kwargs) -> List:
    results = [ t.xlogy(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def xlogy_(self, *args, **kwargs) -> List:
    results = [ t.xlogy_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def zero_(self, *args, **kwargs) -> List:
    results = [ t.zero_(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __str__(self, *args, **kwargs) -> List:
    results = [ t.__str__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __len__(self, *args, **kwargs) -> List:
    results = [ t.__len__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __add__(self, *args, **kwargs) -> List:
    results = [ t.__add__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __call__(self, *args, **kwargs) -> List:
    results = [ t.__call__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __add__(self, *args, **kwargs) -> List:
    results = [ t.__add__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __sub__(self, *args, **kwargs) -> List:
    results = [ t.__sub__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __mul__(self, *args, **kwargs) -> List:
    results = [ t.__mul__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __pow__(self, *args, **kwargs) -> List:
    results = [ t.__pow__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __truediv__(self, *args, **kwargs) -> List:
    results = [ t.__truediv__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __floordiv__(self, *args, **kwargs) -> List:
    results = [ t.__floordiv__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __mod__(self, *args, **kwargs) -> List:
    results = [ t.__mod__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __lshift__(self, *args, **kwargs) -> List:
    results = [ t.__lshift__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __rshift__(self, *args, **kwargs) -> List:
    results = [ t.__rshift__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __and__(self, *args, **kwargs) -> List:
    results = [ t.__and__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __or__(self, *args, **kwargs) -> List:
    results = [ t.__or__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __xor__(self, *args, **kwargs) -> List:
    results = [ t.__xor__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __invert__(self, *args, **kwargs) -> List:
    results = [ t.__invert__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __lt__(self, *args, **kwargs) -> List:
    results = [ t.__lt__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __le__(self, *args, **kwargs) -> List:
    results = [ t.__le__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __eq__(self, *args, **kwargs) -> List:
    results = [ t.__eq__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __ne__(self, *args, **kwargs) -> List:
    results = [ t.__ne__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __gt__(self, *args, **kwargs) -> List:
    results = [ t.__gt__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

  def __ge__(self, *args, **kwargs) -> List:
    results = [ t.__ge__(*args, **kwargs) for t in self.getParts() ]
    return self.createObject(results)

