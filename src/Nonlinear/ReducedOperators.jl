function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:NNContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:NNContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end